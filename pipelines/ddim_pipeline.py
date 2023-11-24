# https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_pix2pix_zero.py#L285

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import math
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


@dataclass
class Pix2PixInversionPipelineOutput(BaseOutput):
    latents: List[torch.FloatTensor]
    images: Union[List[PIL.Image.Image], np.ndarray]

### NOTE: most from diffusers/models/cross_attention.py
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

        ### NOTE: we can also modify this part for saving QKV features
        # if attention_probs.requires_grad:
        #     self.attnstore(attention_probs, is_cross, self.place_in_unet)
            # self.attnstore(value, is_cross, self.place_in_unet)
        self.attnstore(attention_probs, is_cross, self.place_in_unet,
                       hidden_states, query, key, value)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
    
class AttentionStore:
    @staticmethod
    def get_empty_store():
        # return {"down": [], "mid": [], "up": []}
        return {
                "down_cross_attn": [], "mid_cross_attn": [], "up_cross_attn": [],
                "down_self_attn": [], "mid_self_attn": [], "up_self_attn": [],
                "down_cross_feat": [], "mid_cross_feat": [], "up_cross_feat": [],
                "down_self_feat": [], "mid_self_feat": [], "up_self_feat": [],
                "down_cross_query": [], "mid_cross_query": [], "up_cross_query": [],
                "down_self_query": [], "mid_self_query": [], "up_self_query": [],
                "down_cross_key": [], "mid_cross_key": [], "up_cross_key": [],
                "down_self_key": [], "mid_self_key": [], "up_self_key": [],
                "down_cross_value": [], "mid_cross_value": [], "up_cross_value": [],
                "down_self_value": [], "mid_self_value": [], "up_self_value": [],
                }

    ### NOTE: huggingface guys modify this code to only save the attention maps with 256
    def __call__(self, attn, is_cross: bool, place_in_unet: str, hidden_states, query, key, value):

        dict_key = f"{place_in_unet}_{'cross' if is_cross else 'self'}_attn"
        self.step_store[dict_key].append(attn)

        dict_key = f"{place_in_unet}_{'cross' if is_cross else 'self'}_feat"
        self.step_store[dict_key].append(hidden_states)

        dict_key = f"{place_in_unet}_{'cross' if is_cross else 'self'}_query"
        self.step_store[dict_key].append(query)

        dict_key = f"{place_in_unet}_{'cross' if is_cross else 'self'}_key"
        self.step_store[dict_key].append(key)
        
        dict_key = f"{place_in_unet}_{'cross' if is_cross else 'self'}_value"
        self.step_store[dict_key].append(value)

        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.curr_step_index += 1
            self.between_steps()

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            # if self.curr_step_index==50:
            #     for key in self.attention_store:
            #         for i in range(len(self.attention_store[key])):
            #             self.attention_store = self.step_store
                        
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
                    
        self.step_store = self.get_empty_store()

    
    def get_average_attention(self, device='cuda'):
        self.cur_step=50.0
        # self.cur_step=1.0
        average_attention = {key: [item.to(device) / self.cur_step for item in self.attention_store[key]] for key in
                             self.attention_store}
        return average_attention
    
    def aggregate_attention(self, from_where: List[str], res: int, is_cross: bool, element_name='attn') -> torch.Tensor:
        """Aggregates the attention across the different layers and heads at the specified resolution."""
        out = []
        num_pixels = res ** 2
        attention_maps = self.get_average_attention()
        for location in from_where:
            for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}_{element_name}"]:
            # for item in attention_maps[location]:
                if item.shape[1] == num_pixels:
                    cross_maps = item.reshape(-1, res, res, item.shape[-1])
                    out.append(cross_maps)
        if len(out)>0:
            out = torch.cat(out, dim=0)
            out = out.sum(0) / out.shape[0]
        return out

    def reset(self):
        self.curr_step_index = 0
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self, attn_res=16):
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.curr_step_index = 0
        # self.attn_res = attn_res

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.preprocess
def preprocess(image):
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, PIL.Image.Image):
        image = [image]

    if isinstance(image[0], PIL.Image.Image):
        w, h = image[0].size
        w, h = map(lambda x: x - x % 8, (w, h))  # resize to integer multiple of 8

        ### NOTE: the lanczos interpolation is here instead of 
        ### https://github.com/pix2pixzero/pix2pix-zero/blob/2f0b3d942b2a824ec06b328a9be39f784c746b54/src/inversion.py#L54
        image = [np.array(i.resize((w, h), resample=PIL_INTERPOLATION["lanczos"]))[None, :] for i in image]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0
        image = torch.from_numpy(image)
    elif isinstance(image[0], torch.Tensor):
        image = torch.cat(image, dim=0)
    return image


def prepare_unet(unet: UNet2DConditionModel):
    """Modifies the UNet (`unet`) to perform Pix2Pix Zero optimizations."""
    # pix2pix_zero_attn_procs = {}
    for name in unet.attn_processors.keys():
        module_name = name.replace(".processor", "")
        module = unet.get_submodule(module_name)
        if "attn2" in name:
            # pix2pix_zero_attn_procs[name] = Pix2PixZeroCrossAttnProcessor(is_pix2pix_zero=True)
            module.requires_grad_(True)
        else:
            # pix2pix_zero_attn_procs[name] = Pix2PixZeroCrossAttnProcessor(is_pix2pix_zero=False)
            module.requires_grad_(False)

    # unet.set_attn_processor(pix2pix_zero_attn_procs)
    return unet



class StableDiffusionDDIMInvPipeline(DiffusionPipeline):
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

    def prepare_image_latents(self, image, batch_size, dtype, device, generator=None):
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        image = image.to(device=device, dtype=dtype)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if isinstance(generator, list):
            init_latents = [
                self.vae.encode(image[i : i + 1]).latent_dist.sample(generator[i]) for i in range(batch_size)
            ]
            init_latents = torch.cat(init_latents, dim=0)
        else:
            init_latents = self.vae.encode(image).latent_dist.sample(generator)

        init_latents = self.vae.config.scaling_factor * init_latents

        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = torch.cat([init_latents], dim=0)

        latents = init_latents

        return latents

    def register_attention_control(self):
        attn_procs = {}
        cross_att_count = 0
        ### NOTE: here we are dealing with attention processors
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
    def _compute_cosine_seg(attention_maps: torch.Tensor,indices_to_alter: List[int], seg_maps=None):
        x_attn = attention_maps[:,:,indices_to_alter].view(-1,len(indices_to_alter)).t()
        seg_maps_ = torch.cat(seg_maps).view(len(seg_maps),-1)
        return 1.0 - (F.cosine_similarity(x_attn, seg_maps_)).mean()
    
    ### NOTE: It is inversion in essence
    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[str] = None,
        image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        cross_attention_guidance_amount: float = 0.1,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        seg_maps=None,
        resolution=64,
    ):
        # 1. Define call parameters
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

        # 3. Preprocess image
        image = preprocess(image)

        # 4. Prepare latent variables
        latents = self.prepare_image_latents(image, batch_size, self.vae.dtype, device, generator)
        init_latents=latents
        # 5. Encode input prompt
        num_images_per_prompt = 1
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
        )

        # 4. Prepare timesteps
        self.inverse_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.inverse_scheduler.timesteps

        # 6. Rejig the UNet so that we can obtain the cross-attenion maps and
        # use them for guiding the subsequent image generation.
        self.unet = prepare_unet(self.unet)
        ### 8. NOTE: set attention storage. Doing similar things as above
        self.attention_store = AttentionStore()
        self.register_attention_control()

        ### NOTE: for saving all latents
        all_latents=[latents]
        # d_ref_t2attn = {} # reference cross attention maps
        attention_maps_list = []
        self_attention_maps_list = []
        cond_embeddings_list = []

        # 7. Denoising loop where we obtain the cross-attention maps.
        num_warmup_steps = len(timesteps) - num_inference_steps * self.inverse_scheduler.order

        with self.progress_bar(total=num_inference_steps ) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.inverse_scheduler.scale_model_input(latent_model_input, t)

                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    ).sample
                
                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t-1 -> x_t
                latents = self.inverse_scheduler.step(noise_pred, t, latents).prev_sample
                all_latents.append(latents.cpu().detach().clone())

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.inverse_scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
        
        self_avg_dict={}
        cross_avg_dict={}
        for element_name in ['attn', 'feat', 'query', 'key', 'value']:
            self_avg_dict[element_name]={}
            cross_avg_dict[element_name]={}

        for attn_size in [8,16,32,64]:
            for element_name in ['attn', 'feat', 'query', 'key', 'value']:
                self_attn_avg = self.attention_store.aggregate_attention(from_where=("up", "down", "mid"), 
                                                                        res=attn_size,is_cross=False,
                                                                        element_name=element_name)
                cross_attn_avg = self.attention_store.aggregate_attention(from_where=("up", "down", "mid"), 
                                                                        res=attn_size,is_cross=True,
                                                                        element_name=element_name)   
                
                self_avg_dict[element_name][attn_size]=self_attn_avg
                cross_avg_dict[element_name][attn_size]=cross_attn_avg

        inverted_latents = latents.detach().clone()

        # 8. Post-processing
        # image = self.decode_latents(latents.detach())
        image = self.decode_latents(init_latents.detach())

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        # 9. Convert to PIL.
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (inverted_latents, image)

        # return Pix2PixInversionPipelineOutput(latents=all_latents, images=image), attention_maps_list
        return all_latents, image, self_avg_dict, cross_avg_dict


    @torch.no_grad()
    def reconstruct(
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
        cross_attention_guidance_amount: float = 0.1,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        cond_embeddings_list=None,
        resolution=64,
    ):
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
        
        self.attention_store = AttentionStore()
        self.register_attention_control()
        attention_maps_list=[]
        self_attention_maps_list=[]
        # 7. Denoising loop where we obtain the cross-attention maps.
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                _prompt_embeds_ = prompt_embeds
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=_prompt_embeds_,
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
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
        
        ### NOTE: save in dict
        self_avg_dict={}
        cross_avg_dict={}
        for element_name in ['attn', 'feat', 'query', 'key', 'value']:
            self_avg_dict[element_name]={}
            cross_avg_dict[element_name]={}

        for attn_size in [8,16,32,64]:
            for element_name in ['attn', 'feat', 'query', 'key', 'value']:
                self_attn_avg = self.attention_store.aggregate_attention(from_where=("up", "down", "mid"), 
                                                                        res=attn_size,is_cross=False,
                                                                        element_name=element_name)
                cross_attn_avg = self.attention_store.aggregate_attention(from_where=("up", "down", "mid"), 
                                                                        res=attn_size,is_cross=True,
                                                                        element_name=element_name)   
                
                self_avg_dict[element_name][attn_size]=self_attn_avg
                cross_avg_dict[element_name][attn_size]=cross_attn_avg
        # make the reference image (reconstruction)
        image_rec = self.numpy_to_pil(self.decode_latents(latents.detach()))
        return image_rec, self_avg_dict, cross_avg_dict
    