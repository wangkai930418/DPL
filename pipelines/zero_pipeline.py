# https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_pix2pix_zero.py#L285
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


@dataclass
class Pix2PixInversionPipelineOutput(BaseOutput):
    # latents: torch.FloatTensor
    latents: List[torch.FloatTensor]
    images: Union[List[PIL.Image.Image], np.ndarray]


EXAMPLE_DOC_STRING = """"""

EXAMPLE_INVERT_DOC_STRING = """"""


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
    pix2pix_zero_attn_procs = {}
    for name in unet.attn_processors.keys():
        module_name = name.replace(".processor", "")
        module = unet.get_submodule(module_name)
        if "attn2" in name:
            pix2pix_zero_attn_procs[name] = Pix2PixZeroCrossAttnProcessor(is_pix2pix_zero=True)
            module.requires_grad_(True)
        else:
            pix2pix_zero_attn_procs[name] = Pix2PixZeroCrossAttnProcessor(is_pix2pix_zero=False)
            module.requires_grad_(False)

    unet.set_attn_processor(pix2pix_zero_attn_procs)
    return unet


class Pix2PixZeroL2Loss:
    def __init__(self):
        self.loss = 0.0

    def compute_loss(self, predictions, targets):
        self.loss += ((predictions - targets) ** 2).sum((1, 2)).mean(0)


class Pix2PixZeroCrossAttnProcessor:
    """An attention processor class to store the attention weights.
    In Pix2Pix Zero, it happens during computations in the cross-attention blocks."""

    def __init__(self, is_pix2pix_zero=False):
        self.is_pix2pix_zero = is_pix2pix_zero
        if self.is_pix2pix_zero:
            self.reference_cross_attn_map = {}

    def __call__(
        self,
        attn: CrossAttention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        timestep=None,
        loss=None,
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.cross_attention_norm:
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        ### NOTE:  new bookkeeping to save the attn probs
        attn.attn_probs = attention_probs
        if self.is_pix2pix_zero and timestep is not None:
            # new bookkeeping to save the attention weights.
            if loss is None:
                self.reference_cross_attn_map[timestep.item()] = attention_probs.detach().cpu()
            # compute loss
            elif loss is not None:
                prev_attn_probs = self.reference_cross_attn_map.pop(timestep.item())
                loss.compute_loss(attention_probs, prev_attn_probs.to(attention_probs.device))

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class StableDiffusionPix2PixZeroPipeline(DiffusionPipeline):
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
    def generate_caption(self, images):
        """Generates caption for a given image."""
        text = "a photography of"

        prev_device = self.caption_generator.device

        device = self._execution_device
        inputs = self.caption_processor(images, text, return_tensors="pt").to(
            device=device, dtype=self.caption_generator.dtype
        )
        self.caption_generator.to(device)
        outputs = self.caption_generator.generate(**inputs, max_new_tokens=128)

        # offload caption generator
        self.caption_generator.to(prev_device)

        caption = self.caption_processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return caption

    def construct_direction(self, embs_source: torch.Tensor, embs_target: torch.Tensor):
        """Constructs the edit direction to steer the image generation process semantically."""
        return (embs_target.mean(0) - embs_source.mean(0)).unsqueeze(0)

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

    def auto_corr_loss(self, hidden_states, generator=None):
        batch_size, channel, height, width = hidden_states.shape
        if batch_size > 1:
            raise ValueError("Only batch_size 1 is supported for now")

        hidden_states = hidden_states.squeeze(0)
        # hidden_states must be shape [C,H,W] now
        reg_loss = 0.0
        for i in range(hidden_states.shape[0]):
            noise = hidden_states[i][None, None, :, :]
            while True:
                roll_amount = torch.randint(noise.shape[2] // 2, (1,), generator=generator).item()
                reg_loss += (noise * torch.roll(noise, shifts=roll_amount, dims=2)).mean() ** 2
                reg_loss += (noise * torch.roll(noise, shifts=roll_amount, dims=3)).mean() ** 2

                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        return reg_loss

    def kl_divergence(self, hidden_states):
        mean = hidden_states.mean()
        var = hidden_states.var()
        return var + mean**2 - 1 - torch.log(var + 1e-7)
    
    # @replace_example_docstring(EXAMPLE_DOC_STRING)
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
        cross_attention_guidance_amount: float = 0.1,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        # 0. Define the spatial resolutions.
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            image,
            source_embeds,
            target_embeds,
            callback_steps,
            prompt_embeds,
        )

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
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
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
        latents_init = latents.clone()

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 8. Rejig the UNet so that we can obtain the cross-attenion maps and
        # use them for guiding the subsequent image generation.
        self.unet = prepare_unet(self.unet)

        # 7. Denoising loop where we obtain the cross-attention maps.
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs={"timestep": t},
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

        # 8. Compute the edit directions.
        edit_direction = self.construct_direction(source_embeds, target_embeds).to(prompt_embeds.device)

        # 9. Edit the prompt embeddings as per the edit directions discovered.
        prompt_embeds_edit = prompt_embeds.clone()
        prompt_embeds_edit[1:2] += edit_direction

        # 10. Second denoising loop to generate the edited image.
        latents = latents_init
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # we want to learn the latent such that it steers the generation
                # process towards the edited direction, so make the make initial
                # noise learnable
                x_in = latent_model_input.detach().clone()
                x_in.requires_grad = True

                # optimizer
                opt = torch.optim.SGD([x_in], lr=cross_attention_guidance_amount)

                with torch.enable_grad():
                    # initialize loss
                    loss = Pix2PixZeroL2Loss()

                    # predict the noise residual
                    noise_pred = self.unet(
                        x_in,
                        t,
                        encoder_hidden_states=prompt_embeds_edit.detach(),
                        cross_attention_kwargs={"timestep": t, "loss": loss},
                    ).sample

                    loss.loss.backward(retain_graph=False)
                    opt.step()

                # recompute the noise
                noise_pred = self.unet(
                    x_in.detach(),
                    t,
                    encoder_hidden_states=prompt_embeds_edit,
                    cross_attention_kwargs={"timestep": None},
                ).sample

                latents = x_in.detach().chunk(2)[0]

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

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (edited_image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=edited_image, nsfw_content_detected=has_nsfw_concept)
    
    # @replace_example_docstring(EXAMPLE_INVERT_DOC_STRING)
    @torch.no_grad()
    def invert(
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

        ### NOTE: for saving all latents
        all_latents=[latents]
        # d_ref_t2attn = {} # reference cross attention maps
        attention_maps_list = []

        # 7. Denoising loop where we obtain the cross-attention maps.
        num_warmup_steps = len(timesteps) - num_inference_steps * self.inverse_scheduler.order
        # with self.progress_bar(total=num_inference_steps - 2) as progress_bar:
            # for i, t in enumerate(timesteps[1:-1]):
            ### ATTN: above is the original code, 
            # I think it's incorrect and modify by referring to other repos, 
            # also the scheduler.py code
        with self.progress_bar(total=num_inference_steps ) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.inverse_scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs={"timestep": t},
                ).sample

                # NOTE: visualize DDIM inversion cross attention
                all_attentions=[]
                res=16
                for name, module in self.unet.named_modules():
                    module_name = type(module).__name__
                    if module_name == "CrossAttention" and 'attn2' in name:
                        curr = module.attn_probs
                        if curr.shape[-1] == 77 and curr.shape[-2] == (res**2):
                            all_attentions.append(curr)

                attention_maps = torch.cat(all_attentions, dim=0).mean(0)
                attention_maps =  attention_maps.reshape(res,res,-1)
                attention_maps_list.append(attention_maps.detach().cpu())

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

        inverted_latents = latents.detach().clone()

        # 8. Post-processing
        image = self.decode_latents(latents.detach())

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        # 9. Convert to PIL.
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (inverted_latents, image)

        # return Pix2PixInversionPipelineOutput(latents=inverted_latents, images=image)
        return Pix2PixInversionPipelineOutput(latents=all_latents, images=image), attention_maps_list


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
    ):
        # 0. Define the spatial resolutions.
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        # self.check_inputs(
        #     prompt,
        #     image,
        #     source_embeds,
        #     target_embeds,
        #     callback_steps,
        #     prompt_embeds,
        # )

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

        # 7. Denoising loop where we obtain the cross-attention maps.
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs={"timestep": t},
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

        # make the reference image (reconstruction)
        image_rec = self.numpy_to_pil(self.decode_latents(latents.detach()))
        return image_rec