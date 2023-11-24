import math
import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import PIL
import torch 
import torch.nn.functional as F
from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
    CLIPFeatureExtractor,
    CLIPTextModel,
    CLIPTokenizer,
)

from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.models.cross_attention import CrossAttention
from diffusers.schedulers import DDIMScheduler, DDPMScheduler, EulerAncestralDiscreteScheduler, LMSDiscreteScheduler
from diffusers.schedulers.scheduling_ddim_inverse import DDIMInverseScheduler
from diffusers.utils import (
    PIL_INTERPOLATION,
    BaseOutput,
    is_accelerate_available,
    is_accelerate_version,
    logging,
    randn_tensor,
    replace_example_docstring,
)
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def prepare_unet(unet: UNet2DConditionModel):
    """Modifies the UNet (`unet`) to perform Pix2Pix Zero optimizations."""
    # pix2pix_zero_attn_procs = {}
    for name in unet.attn_processors.keys():
        module_name = name.replace(".processor", "")
        module = unet.get_submodule(module_name)
        if "attn2" in name:
            module.requires_grad_(True)
        else:
            module.requires_grad_(False)

    return unet

### ========================= Copy from attend and excite pipeline =========================
### https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_attend_and_excite.py

class AttentionStore:
    @staticmethod
    def get_empty_store():
        return {"down": [], "mid": [], "up": []}

    ### NOTE: huggingface guys modify this code to only save the attention maps with 256
    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= 0 and is_cross:
            if attn.shape[1] == self.attn_res**2:
                self.step_store[place_in_unet].append(attn)

        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.between_steps()

    def between_steps(self):
        self.attention_store = self.step_store
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = self.attention_store
        return average_attention

    def aggregate_attention(self, from_where: List[str]) -> torch.Tensor:
        """Aggregates the attention across the different layers and heads at the specified resolution."""
        out = []
        attention_maps = self.get_average_attention()
        for location in from_where:
            for item in attention_maps[location]:
                cross_maps = item.reshape(-1, self.attn_res, self.attn_res, item.shape[-1])
                out.append(cross_maps)
        out = torch.cat(out, dim=0)
        out = out.sum(0) / out.shape[0]
        return out

    def reset(self):
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self, attn_res=16):
        """
        Initialize an empty AttentionStore :param step_index: used to visualize only a specific step in the diffusion
        process
        """
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.curr_step_index = 0
        self.attn_res = attn_res


class AttendExciteCrossAttnProcessor:
    def __init__(self, attnstore, place_in_unet):
        super().__init__()
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet

    def __call__(
        self, 
        attn: CrossAttention, 
        hidden_states, 
        encoder_hidden_states=None, 
        attention_mask=None
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # only need to store attention maps during the Attend and Excite process
        if attention_probs.requires_grad:
            self.attnstore(attention_probs, is_cross, self.place_in_unet)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
    


### ========================= ORIGINAL zero-shot image translation processor =========================

class StableDiffusion_DetPipeline(DiffusionPipeline):
    _optional_components = [
        "safety_checker",
        "feature_extractor",
        "caption_generator",
        "caption_processor",
        "inverse_scheduler",
    ]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[DDPMScheduler, DDIMScheduler, EulerAncestralDiscreteScheduler, LMSDiscreteScheduler],
        feature_extractor: CLIPFeatureExtractor,
        safety_checker: StableDiffusionSafetyChecker,
        inverse_scheduler: DDIMInverseScheduler,
        caption_generator: BlipForConditionalGeneration,
        caption_processor: BlipProcessor,
        requires_safety_checker: bool = True,
    ):
        super().__init__()

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            caption_processor=caption_processor,
            caption_generator=caption_generator,
            inverse_scheduler=inverse_scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_sequential_cpu_offload
    def enable_sequential_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
        text_encoder, vae and safety checker have their state dicts saved to CPU and then are moved to a
        `torch.device('meta') and loaded to GPU only when their specific submodule has its `forward` method called.
        Note that offloading happens on a submodule basis. Memory savings are higher than with
        `enable_model_cpu_offload`, but performance is lower.
        """
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            cpu_offload(cpu_offloaded_model, device)

        if self.safety_checker is not None:
            cpu_offload(self.safety_checker, execution_device=device, offload_buffers=True)

    def enable_model_cpu_offload(self, gpu_id=0):
        if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError("`enable_model_offload` requires `accelerate v0.17.0` or higher.")

        device = torch.device(f"cuda:{gpu_id}")

        hook = None
        for cpu_offloaded_model in [self.vae, self.text_encoder, self.unet, self.vae]:
            _, hook = cpu_offload_with_hook(cpu_offloaded_model, device, prev_module_hook=hook)

        if self.safety_checker is not None:
            _, hook = cpu_offload_with_hook(self.safety_checker, device, prev_module_hook=hook)

        # We'll offload the last model manually.
        self.final_offload_hook = hook

    @property
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._execution_device
    def _execution_device(self):
        if not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._encode_prompt
    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker
    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        else:
            has_nsfw_concept = None
        return image, has_nsfw_concept

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents
    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        image,
        source_embeds,
        target_embeds,
        callback_steps,
        prompt_embeds=None,
    ):
        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )
        if source_embeds is None and target_embeds is None:
            raise ValueError("`source_embeds` and `target_embeds` cannot be undefined.")

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

    #  Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @torch.no_grad()
    def get_embeds(self, prompt: List[str], batch_size: int = 16) -> torch.FloatTensor:
        num_prompts = len(prompt)
        embeds = []
        for i in range(0, num_prompts, batch_size):
            prompt_slice = prompt[i : i + batch_size]

            input_ids = self.tokenizer(
                prompt_slice,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids

            input_ids = input_ids.to(self.text_encoder.device)
            embeds.append(self.text_encoder(input_ids)[0])

        return torch.cat(embeds, dim=0).mean(0)[None]

    def register_attention_control(self):
        attn_procs = {}
        cross_att_count = 0
        for name in self.unet.attn_processors.keys():
            if name.startswith("mid_block"):
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                place_in_unet = "down"
            else:
                continue

            cross_att_count += 1
            attn_procs[name] = AttendExciteCrossAttnProcessor(
                attnstore=self.attention_store, place_in_unet=place_in_unet
            )

        self.unet.set_attn_processor(attn_procs)
        self.attention_store.num_att_layers = cross_att_count
    
    @staticmethod    
    def _compute_cosine_adj(attention_maps: torch.Tensor,indices_to_alter: List[int], adj_indices_to_alter: List[int],):
        x_attn = attention_maps[:,:,indices_to_alter].view(-1,len(indices_to_alter)).t()
        x_attn_adj = attention_maps[:,:,adj_indices_to_alter].view(-1,len(adj_indices_to_alter)).t()

        return 1.0 - (F.cosine_similarity(x_attn, x_attn_adj)).mean()
    
    @staticmethod    
    def _compute_cosine_seg(attention_maps: torch.Tensor,indices_to_alter: List[int], seg_maps=None):
        x_attn = attention_maps[:,:,indices_to_alter].view(-1,len(indices_to_alter)).t()
        seg_maps_ = torch.cat(seg_maps).view(len(seg_maps),-1)

        return (F.cosine_similarity(x_attn, 1-seg_maps_)).mean()
    
    @staticmethod    
    def _compute_IoU_loss(attention_maps: torch.Tensor,indices_to_alter: List[int], seg_maps=None):
        x_attn = attention_maps[:,:,indices_to_alter].view(-1,len(indices_to_alter)).t()
        seg_maps_ = torch.cat(seg_maps).view(len(seg_maps),-1)
        
        comple_seg_maps_ = 1.0 - seg_maps_
        length=len(seg_maps_)
        loss_list=[(x_attn[i]*comple_seg_maps_[i]).sum()/x_attn[i].sum() for i in range(length)]

        return sum(loss_list)/float(length)
    
    @staticmethod    
    def _compute_cosine(attention_maps: torch.Tensor,indices_to_alter: List[int],):
        x_attn = attention_maps[:,:,indices_to_alter].view(-1,len(indices_to_alter)).t()
        cos_mask = torch.tril(torch.ones((len(indices_to_alter),len(indices_to_alter))),diagonal=-1).bool()
        ### NOTE: broadcasting
        cos_sim = F.cosine_similarity(x_attn[:,:,None], x_attn.t()[None,:,:])
        cos_dist = cos_sim[cos_mask].mean()
        return cos_dist
    
    def _aggregate_and_get_max_attention_per_token(
        self,
        indices: List[int],
        smooth_op = True,
        softmax_op = True,
    ):
        """Aggregates the attention for each token and computes the max activation value for each token to alter."""
        attention_maps = self.attention_store.aggregate_attention(
            from_where=("up", "down", "mid"),
        )
        # attention_maps_cp = attention_maps.clone()

        max_attention_per_index = self._compute_max_attention_per_index(
            attention_maps=attention_maps,
            indices=indices,
            smooth_op = smooth_op,
        )

        return max_attention_per_index, attention_maps
    
    @staticmethod
    def _compute_max_attention_per_index(
        attention_maps: torch.Tensor,
        indices: List[int],
        smooth_op = True,
    ) -> List[torch.Tensor]:
        """Computes the maximum attention value for each of the tokens we wish to alter."""
        attention_for_text = attention_maps[:, :, 1:-1]
        
        # attention_for_text *= 100
        ### NOTE: the above inplace operator is not pleasant
        attention_for_text = attention_for_text * 100
        attention_for_text = torch.nn.functional.softmax(attention_for_text, dim=-1)

        # Shift indices since we removed the first token
        indices = [index - 1 for index in indices]

        # Extract the maximum values
        max_indices_list = []
        for i in indices:
            image = attention_for_text[:, :, i]
            if smooth_op:
                smoothing = GaussianSmoothing().to(attention_maps.device)
                input = F.pad(image.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="reflect")
                image = smoothing(input).squeeze(0).squeeze(0)
            max_indices_list.append(image.max())
        return max_indices_list
    


    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        image: Optional[Union[torch.FloatTensor, PIL.Image.Image]] = None,
        source_embeds: torch.Tensor = None,
        target_embeds: torch.Tensor = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        ### NOTE: below are new
        all_latents=None,          
        null_inner_steps=10,
        attn_inner_steps=10,
        print_freq=5,
        token_indices = [2, 5],
        placeholder_token_id=49408,
        index_no_updates=None,
        alpha_cos = 72.0,
        alpha_iou = 72.0,
        alpha_kl = 72.0,
        alpha_sim = 72.0,
        alpha_adj = 72.0,
        beta_cos = 0.1,
        beta_iou = 0.1,
        beta_kl = 0.1,
        beta_sim = 0.1,
        beta_adj = 0.1,
        attn_res = 16, 
        smooth_op = True,  
        softmax_op = True, 
        seg_maps=None,
        loss_type = 'cosine',
        lam_cos=0.0,
        lam_iou=0.0,
        lam_kl=0.0,
        lam_sim=0.0,
        lam_adj=0.0,
        adj_indices_to_alter=None
    ):
        ### NOTE: lower the cuda usage
        # self.vae.to('cpu')
        
        indices_to_alter = token_indices
        # 0. Define the spatial resolutions.
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 3. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        if cross_attention_kwargs is None:
            cross_attention_kwargs = {}

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        ### NOTE: get input ids for repeat usage
        text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
        text_input_ids = text_inputs.input_ids

        # 3. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Generate the inverted noise from the input image or any other image
        # generated from the input prompt.
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 8. Rejig the UNet so that we can obtain the cross-attenion maps and
        # use them for guiding the subsequent image generation.
        self.unet = prepare_unet(self.unet)

        ### 8. NOTE: set attention storage. Doing similar things as above
        self.attention_store = AttentionStore(attn_res=attn_res)
        self.register_attention_control()

        orig_embeds_params = (self.text_encoder).get_input_embeddings().weight.data.clone()

        # 10. Second denoising loop to generate the edited image.
        # latents = latents_init
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        ### NOTE: set a hint to remind if the optimization is done
        uncond_embeddings_list = []
        cond_embeddings_list = []
        attention_maps_list = []

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                ### ============== 1st part START: ATTENTION ==============
                # cos_target_loss = 1.0- alpha_cos - (1.0 - beta_cos - alpha_cos)/50*i 
                # cos_target_loss = alpha_cos - (alpha_cos - beta_cos)/ float(num_inference_steps) * i 
                cos_target_loss = np.exp(-float(i)/alpha_cos)*beta_cos
                iou_target_loss = np.exp(-float(i)/alpha_iou)*beta_iou
                sim_target_loss = np.exp(-float(i)/alpha_sim)*beta_sim
                adj_target_loss = np.exp(-float(i)/alpha_adj)*beta_adj

                target_string=f"\n Step {i} Target cos: {cos_target_loss:0.6f}; "+ \
                                f"iou: {iou_target_loss:0.6f}; mutual sim: {sim_target_loss:0.6f};" + \
                                    f"adj: {adj_target_loss:0.6f};"
                
                print(target_string)     

                with torch.enable_grad():
                    ### NOTE: Initialize the optimizer and update parameters
                    cond_optim = torch.optim.AdamW(self.text_encoder.get_input_embeddings().parameters())
                    for j in range(attn_inner_steps):
                        encoder_hidden_states = self.text_encoder(text_input_ids.to(device))[0].to(dtype=torch.float32)
                        noise_pred = self.unet(latents,
                                            t,
                                            encoder_hidden_states=encoder_hidden_states,
                                            cross_attention_kwargs=cross_attention_kwargs,
                                            ).sample
                        self.unet.zero_grad()
                        # Get max activation value for each subject token
                        _, attention_maps = self._aggregate_and_get_max_attention_per_token(
                                                                                    indices=token_indices,
                                                                                    smooth_op=smooth_op,
                                                                                    softmax_op=softmax_op)
                        if (lam_cos+lam_iou) == 0.0:
                            # print('no need to backpropagate')
                            self.unet.zero_grad()
                            torch.cuda.empty_cache()
                            break

                        if lam_cos > 0.0:
                            cosine_loss = self._compute_cosine_seg(attention_maps, indices_to_alter, seg_maps)    
                        else:
                            cosine_loss = torch.Tensor([0.0]).cuda()

                        if lam_iou > 0.0:
                            iou_loss = self._compute_IoU_loss(attention_maps, indices_to_alter, seg_maps)    
                        else:
                            iou_loss = torch.Tensor([0.0]).cuda()

                        if lam_adj > 0.0 and (adj_indices_to_alter is not None):
                            adj_loss = self._compute_cosine_adj(attention_maps, indices_to_alter, adj_indices_to_alter)    
                        else:
                            adj_loss = torch.Tensor([0.0]).cuda()
                            
                        if lam_sim > 0.0 and len(indices_to_alter)>1:
                            sim_loss = self._compute_cosine(attention_maps, indices_to_alter)    
                        else:
                            sim_loss = torch.Tensor([0.0]).cuda()

                        loss = cosine_loss*lam_cos + iou_loss*lam_iou +  adj_loss*lam_adj + sim_loss*lam_sim

                        print_string = f"Step {i}, Attend {j} | Loss:{loss.item():0.6f};" + \
                                         f"cosine: {cosine_loss.item():0.6f}; " + \
                                            f"iou: {iou_loss.item():0.6f}; " + \
                                                f"adj: {adj_loss.item():0.6f};"  + \
                                                    f"inter sim: {sim_loss.item():0.6f}" 

                        if j%print_freq ==0:
                            print(print_string)
                            
                        if (cosine_loss < cos_target_loss and iou_loss < iou_target_loss and adj_loss < adj_target_loss ):
                            if j%print_freq !=0:
                                print(print_string)

                        # # NOTE:Let's make sure we don't update any embedding weights besides the newly added token
                        loss.backward(retain_graph=False)
                        if not (cosine_loss < cos_target_loss and iou_loss < iou_target_loss and adj_loss < adj_target_loss ):
                            cond_optim.step()

                        cond_optim.zero_grad()

                        if not (cosine_loss < cos_target_loss and iou_loss < iou_target_loss and adj_loss < adj_target_loss ):
                            with torch.no_grad():
                                self.text_encoder.get_input_embeddings().weight[index_no_updates] = orig_embeds_params[index_no_updates]
                        
                        if (cosine_loss < cos_target_loss and iou_loss < iou_target_loss and adj_loss < adj_target_loss ):
                            break

                ### ============== 1st part END: ATTENTION ==============
                torch.cuda.empty_cache()
                cond_embeds = self.text_encoder.get_input_embeddings().weight[placeholder_token_id]
                cond_embeddings_list.append(cond_embeds.detach().cpu())
                attention_maps_list.append(attention_maps.detach().cpu())

                prompt_embeds = self._encode_prompt(
                        prompt,
                        device,
                        num_images_per_prompt,
                        do_classifier_free_guidance,
                        negative_prompt,
                        prompt_embeds=None, ### NOTE: reinitialize
                        negative_prompt_embeds=negative_prompt_embeds,
                    )
                
                ### ============== 2nd part START: NULL INVERSION ==============
                uncond_embeddings, cond_embeddings = prompt_embeds.chunk(2)
                latent_prev = all_latents[len(all_latents) - i - 2]
                
                uncond_embeddings = uncond_embeddings.detach().clone().requires_grad_(True)
                cond_embeddings = cond_embeddings.detach().clone().requires_grad_(False)
                opt = torch.optim.Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
                
                with torch.enable_grad():
                    for j in range(null_inner_steps):
                        context=torch.cat([uncond_embeddings, cond_embeddings])
                        self.unet.zero_grad()
                        ### NOTE: this line might be the reason for retain_graph True, since some cache not released with backward()
                        # with torch.autocast(device_type='cuda', dtype=torch.float16):
                        noise_pred = self.unet(latent_model_input,
                                            t,
                                            encoder_hidden_states=context,
                                            cross_attention_kwargs=cross_attention_kwargs,
                                            ).sample
                        ### NOTE: consider modify the above for loss scaler
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                        latents_prev_rec = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                        loss = F.mse_loss(latents_prev_rec, latent_prev)

                        loss.backward(retain_graph=False)
                        opt.step()
                        opt.zero_grad()
                                
                        if j % print_freq == 0:
                            print(f'Step {i}, Null text loop {j} Loss: {loss.item():0.6f}')
                            
                ### ============== 2nd part END: NULL INVERSION ==============
                torch.cuda.empty_cache()
                prompt_embeds=torch.cat([uncond_embeddings, cond_embeddings])
                uncond_embeddings_list.append(uncond_embeddings.cpu().detach())

                ### NOTE: this line might be the reason for retain_graph True, since some cache not released with backward()
                with torch.no_grad():
                    noise_pred = self.unet( 
                                    latent_model_input,
                                    t,
                                    encoder_hidden_states=prompt_embeds,
                                    cross_attention_kwargs=cross_attention_kwargs,
                                    ).sample
                
                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        # 11. Post-process the latents.
        edited_image = self.decode_latents(latents)

        # 12. Run the safety checker.
        edited_image, has_nsfw_concept = self.run_safety_checker(edited_image, device, prompt_embeds.dtype)

        # 13. Convert to PIL.
        if output_type == "pil":
            edited_image = self.numpy_to_pil(edited_image)

        return edited_image, attention_maps_list, uncond_embeddings_list, cond_embeddings_list
    

    @torch.no_grad()
    def reconstruct(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        all_latents=None,  
        token_indices = [2, 5],
        placeholder_token_id=49408,
        max_iter_to_alter=25,
        index_no_updates=None,
        cond_embeddings_list=None,
        uncond_embeddings_list=None,
    ):
        indices_to_alter = token_indices
        # 0. Define the spatial resolutions.
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 3. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        if cross_attention_kwargs is None:
            cross_attention_kwargs = {}

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        ### NOTE: get input ids for repeat usage
        text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
        text_input_ids = text_inputs.input_ids

        # 3. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Generate the inverted noise from the input image or any other image
        # generated from the input prompt.
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 8. Rejig the UNet so that we can obtain the cross-attenion maps and
        # use them for guiding the subsequent image generation.
        # 10. Second denoising loop to generate the edited image.
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        ### NOTE: set a hint to remind if the optimization is done
        # uncond_embeddings_list = []
        # cond_embeddings_list = []
        # attention_maps_list = []

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                ### NOTE: replace the cond and unconditional embeddings here
                token_embeds = self.text_encoder.get_input_embeddings().weight.data
                for ind in range(len(placeholder_token_id)):
                    token_embeds[placeholder_token_id[ind]] = cond_embeddings_list[i][ind]
                # self.text_encoder.get_input_embeddings().weight[index_no_updates] = orig_embeds_params[index_no_updates]
                encoder_hidden_states = self.text_encoder(text_input_ids.to(device))[0].to(dtype=torch.float32)
                
                prompt_embeds = torch.cat([uncond_embeddings_list[i].cuda(), encoder_hidden_states.cuda()])
                
                noise_pred = self.unet( 
                                    latent_model_input,
                                    t,
                                    encoder_hidden_states=prompt_embeds,
                                    cross_attention_kwargs=cross_attention_kwargs,
                                    ).sample
                
                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        # 11. Post-process the latents.
        edited_image = self.decode_latents(latents)

        # 12. Run the safety checker.
        edited_image, has_nsfw_concept = self.run_safety_checker(edited_image, device, prompt_embeds.dtype)

        # 13. Convert to PIL.
        if output_type == "pil":
            edited_image = self.numpy_to_pil(edited_image)

        return edited_image
    




### NOTE: gaussian smoothing from attend and excite

class GaussianSmoothing(torch.nn.Module):
    # channels=1, kernel_size=kernel_size, sigma=sigma, dim=2
    def __init__(
        self,
        channels: int = 1,
        kernel_size: int = 3,
        sigma: float = 0.5,
        dim: int = 2,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, float):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-(((mgrid - mean) / (2 * std)) ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer("weight", kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError("Only 1, 2 and 3 dimensions are supported. Received {}.".format(dim))

    def forward(self, input):
        return self.conv(input, weight=self.weight.to(input.dtype), groups=self.groups)