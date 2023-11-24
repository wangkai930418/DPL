import math
import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union
import copy
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



def entropy(p):
        return -(p * p.log()).sum() 

### ========================= initially Copied from attend and excite pipeline =========================
### https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_attend_and_excite.py

class AttentionStore:
    @staticmethod
    def get_empty_store():
        return {"down": [], "mid": [], "up": []}

    ### NOTE: huggingface guys modify this code to only save the attention maps with 16*16
    def __call__(self, attn, is_cross: bool, place_in_unet: str, 
                        cross_replace_steps, self_replace_steps):
        
        if self.cur_att_layer >= 0 and is_cross:
            if attn.shape[1] == self.attn_res**2:
                self.step_store[place_in_unet].append(attn)

        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.between_steps()

        ### NOTE: cross-attention modification
        # # if is_cross:
        # if is_cross and self.curr_step_index < cross_replace_steps:
        #     C, H, W = attn.shape
        #     attn_ = attn.reshape(4, C//4, H, W)
        #     attn_[3] = attn_[2]
        #     return attn_.reshape(C,H,W)
        
        # ### NOTE: self-attention modification
        # elif (not is_cross) and self.curr_step_index < self_replace_steps:
        #     C, H, W = attn.shape
        #     ### NOTE: seems 8*8 attention maps are not influencing
        #     if H <= 256:
        #     # if H == 256:
        #         attn_ = attn.reshape(4, C//4, H, W)
        #         attn_[3] = attn_[2]
        #         return attn_.reshape(C,H,W)
        #     else:
        #         return attn
        # else:
        #     return attn
        return attn

    def between_steps(self):
        self.attention_store = self.step_store
        self.step_store = self.get_empty_store()
        ### NOTE: add two lines to update the steps
        self.curr_step_index+=1

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

    def get_raw_attention(self, from_where: List[str], indice=None) -> torch.Tensor:
        """Aggregates the attention across the different layers and heads at the specified resolution."""
        out1 = []
        out2 = []
        attention_maps = self.get_average_attention()
        for location in from_where:
            for item in attention_maps[location]:
                cross_maps = item.reshape(-1, self.attn_res, self.attn_res, item.shape[-1])

                out1.append(cross_maps[2,:,:,indice].unsqueeze(0).squeeze(-1))
                out2.append(cross_maps[3,:,:,indice].unsqueeze(0).squeeze(-1))
                    
        out1 = torch.cat(out1, dim=0)
        out1 = out1.sum(0) / out1.shape[0]
        
        out2 = torch.cat(out2, dim=0)
        out2 = out2.sum(0) / out2.shape[0]
        
        return torch.cat([out1.unsqueeze(0).unsqueeze(0),out2.unsqueeze(0).unsqueeze(0)])
    
    def reset(self):
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self, attn_res=16):
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.curr_step_index = 0
        self.attn_res = attn_res


class AttentionReplace(AttentionStore):
    def __call__(self, attn, is_cross: bool, place_in_unet: str, 
                        cross_replace_steps, self_replace_steps):
        
        if self.cur_att_layer >= 0 and is_cross:
            if attn.shape[1] == self.attn_res**2:
                self.step_store[place_in_unet].append(attn)

        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.between_steps()

        ### NOTE: cross-attention modification
        # if is_cross:
        if is_cross and self.curr_step_index < cross_replace_steps:
            C, H, W = attn.shape
            attn_ = attn.reshape(4, C//4, H, W)
            
            ### NOTE: amplify
            # breakpoint()
            attn_replace=attn_[2].clone()
            if self.indices_to_amplify is not None:
                for reweight_id in range(len(self.indices_to_amplify)):
                    attn_replace[:,:, self.indices_to_amplify[reweight_id]] *= self.amplify_scale[reweight_id]

            attn_[3] = attn_replace
            # attn_[3] = attn_[2]
            return attn_.reshape(C,H,W)
        
        ### NOTE: self-attention modification
        elif (not is_cross) and self.curr_step_index < self_replace_steps:
            C, H, W = attn.shape
            ### NOTE: seems 8*8 attention maps are not influencing
            if H <= 256:
            # if H == 256:
                attn_ = attn.reshape(4, C//4, H, W)
                attn_[3] = attn_[2]
                return attn_.reshape(C,H,W)
            else:
                return attn
        else:
            return attn
    
    def __init__(self, attn_res=16, mapper=None, alphas=None, indices_to_amplify=None, amplify_scale=1.0,):
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.curr_step_index = 0
        self.attn_res = attn_res
        self.mapper = mapper
        self.alphas = alphas
        self.indices_to_amplify = indices_to_amplify
        self.amplify_scale = amplify_scale


class AttentionRefine(AttentionStore):
    def __call__(self, attn, is_cross: bool, place_in_unet: str, 
                        cross_replace_steps, self_replace_steps):
        
        if self.cur_att_layer >= 0 and is_cross:
            if attn.shape[1] == self.attn_res**2:
                self.step_store[place_in_unet].append(attn)

        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.between_steps()

        ### NOTE: cross-attention modification
        if is_cross and self.curr_step_index < cross_replace_steps:
            C, H, W = attn.shape
            attn_ = attn.reshape(4, C//4, H, W)

            attn_base_replace = attn_[2][:,:,self.mapper]
            attn_replace = attn_[3] * (1-self.alphas) + attn_base_replace * self.alphas

            ### NOTE: amplify
            if self.indices_to_amplify is not None:
                for reweight_id in range(len(self.indices_to_amplify)):
                    attn_replace[:,:, self.indices_to_amplify[reweight_id]] *= self.amplify_scale[reweight_id]

            attn_[3] = attn_replace
            return attn_.reshape(C,H,W)
        
        ### NOTE: self-attention modification
        elif (not is_cross) and self.curr_step_index < self_replace_steps:
            C, H, W = attn.shape
            if H <= 256:
                attn_ = attn.reshape(4, C//4, H, W)
                attn_[3] = attn_[2]
                return attn_.reshape(C,H,W)
            else:
                return attn
        else:
            return attn
        
    def __init__(self, attn_res=16, mapper=None, alphas=None, indices_to_amplify=None, amplify_scale=1.0,):
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.curr_step_index = 0
        self.attn_res = attn_res
        self.mapper = mapper
        self.alphas = alphas
        self.indices_to_amplify = indices_to_amplify
        self.amplify_scale = amplify_scale



class AttendExciteCrossAttnProcessor:
    def __init__(self, attnstore, place_in_unet, cross_replace_steps=40, self_replace_steps=20):
        super().__init__()
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet
        self.cross_replace_steps=cross_replace_steps
        self.self_replace_steps=self_replace_steps

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

        # NOTE: only need to store attention maps during the Attend and Excite process
        ### but may need modification for replacing the attentions
        # if attention_probs.requires_grad:
            # self.attnstore(attention_probs, is_cross, self.place_in_unet)
        attention_probs = self.attnstore(attention_probs, is_cross, self.place_in_unet, 
                                         self.cross_replace_steps, self.self_replace_steps)
        
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
    

class StableDiffusion_EditPipeline(DiffusionPipeline):
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
                attnstore=self.attention_store, place_in_unet=place_in_unet,
                cross_replace_steps=self.cross_replace_steps, 
                self_replace_steps=self.self_replace_steps,
            )

        self.unet.set_attn_processor(attn_procs)
        self.attention_store.num_att_layers = cross_att_count


    def _get_attention_maps(
        self, indices
    ):
        """Aggregates the attention for each token and computes the max activation value for each token to alter."""
        attention_maps = self.attention_store.get_raw_attention(
            from_where=("up", "down", "mid"), indice= indices
        )
        return  attention_maps
    
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        edit_prompt: Optional[Union[str, List[str]]] = None,
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
        refine=False,
        replace=False,
        local=False,
        mapper = None, 
        alphas = None,
        cross_replace_steps=40,
        self_replace_steps=20,
        indices_to_amplify=None,
        amplify_scale=1.0,
        indices_local=[2,5],
    ):
        ### NOTE: hard coding for these definitions
        self.cross_replace_steps = cross_replace_steps
        self.self_replace_steps = self_replace_steps
        
        # indices_to_alter = token_indices
        # 0. Define the spatial resolutions.
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 3. Define call parameters
        # if prompt is not None and isinstance(prompt, str):
        #     batch_size = 1
        # elif prompt is not None and isinstance(prompt, list):
        #     batch_size = len(prompt)
        # else:
        #     batch_size = prompt_embeds.shape[0]

        batch_size = 2

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
        edit_prompt_embeds = self._encode_prompt(
            edit_prompt, ### NOTE: prompt before
            device,
            num_images_per_prompt,
            False,  # NOTE: do_classifier_free_guidance,
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
        
        ### NOTE: duplicate the latents for further usage
        latents=torch.cat([latents] * batch_size) 
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            edit_prompt_embeds.dtype,  # prompt_embeds.dtype, 
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 8. Rejig the UNet so that we can obtain the cross-attenion maps and

        ### 8. NOTE: set attention storage. Doing similar things as above
        if replace:
            self.attention_store = AttentionReplace(attn_res=16, 
                                                    indices_to_amplify=indices_to_amplify, 
                                                    amplify_scale=amplify_scale,)
        elif refine:
            self.attention_store = AttentionRefine(attn_res=16, 
                                                   mapper=mapper, alphas=alphas, 
                                                   indices_to_amplify=indices_to_amplify, 
                                                   amplify_scale=amplify_scale)
        else:
            self.attention_store = AttentionStore(attn_res=16)

        self.register_attention_control()

        # 10. Second denoising loop to generate the edited image.
        # latents = latents_init
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                ### NOTE: replace the cond and unconditional embeddings here
                ### NOTE: the token_embeds are making the text_encoder embeddings totally changed
                token_embeds = self.text_encoder.get_input_embeddings().weight.data
                for ind in range(len(placeholder_token_id)):
                    token_embeds[placeholder_token_id[ind]] = cond_embeddings_list[i][ind]

                encoder_hidden_states = self.text_encoder(text_input_ids.to(device))[0].to(dtype=torch.float32)
                
                ### NOTE: the parameters are changing by timestep
                edit_prompt_embeds = self._encode_prompt(
                    edit_prompt, ### NOTE: prompt before
                    device,
                    num_images_per_prompt,
                    False,  # NOTE: do_classifier_free_guidance,
                    negative_prompt,
                    prompt_embeds=None,
                    negative_prompt_embeds=negative_prompt_embeds,
                    )
                
                if do_classifier_free_guidance:
                    prompt_embeds = torch.cat([uncond_embeddings_list[i].cuda(), 
                                            uncond_embeddings_list[i].cuda(), 
                                            encoder_hidden_states.cuda(),
                                            edit_prompt_embeds,
                                            ])
                else:
                    prompt_embeds = torch.cat([
                                            encoder_hidden_states.cuda(),
                                            edit_prompt_embeds,
                                            ])
                    
                noise_pred = self.unet(
                                    latent_model_input,
                                    t,
                                    encoder_hidden_states=prompt_embeds,
                                    cross_attention_kwargs=cross_attention_kwargs,
                                    ).sample
                
                if local:
                    attention_maps = self._get_attention_maps(indices=indices_local)
                    
                    ### NOTE: get mask for local edit
                    mask = []
                    # for index in range(len(attention_maps)):
                    temp=F.max_pool2d(attention_maps,(3,3), (1,1), (1,1))
                    temp=F.interpolate(temp, size=(64,64))
                    temp = temp / temp.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
                    temp = temp.gt(0.3)
                    ### NOTE: threshold 0.3 hard code
                    # breakpoint()
                    mask = temp[:1] + temp
                else:
                    mask=None

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                if local:
                    x_t = latents
                    x_t = x_t[:1] + mask * (x_t - x_t[:1])
                    latents = x_t
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