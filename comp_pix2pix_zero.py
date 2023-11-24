import torch
import PIL.Image as Image   
from diffusers import DDIMScheduler, DDIMInverseScheduler
from pipelines.pix2pix_zero_pipeline import StableDiffusionPix2PixZeroPipeline
import PIL.Image as Image   
from diffusers.utils import pt_to_pil, numpy_to_pil
import numpy as np
import argparse
import os
import copy

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=str, default=None)
    parser.add_argument('--results_folder', type=str, default=None)
    parser.add_argument('--results_folder_edit', type=str, default=None)
    
    parser.add_argument('--num_ddim_steps', type=int, default=50)
    parser.add_argument('--model_path', type=str, default='CompVis/stable-diffusion-v1-4')
    parser.add_argument('--use_float_16', action='store_true')

    parser.add_argument('--prompt_str', type=str, default=None)

    parser.add_argument('--prompt_file', type=str, default=None)

    parser.add_argument('--original_words', nargs='+', type=str, default=None)
    parser.add_argument('--replace_words', nargs='+', type=str, default=None)

    args = parser.parse_args()
    return args


if __name__=="__main__":
    args = arguments()
    print(args)
    torch_dtype = torch.float32

    pipeline = StableDiffusionPix2PixZeroPipeline.from_pretrained(
        args.model_path, conditions_input_image=False, torch_dtype=torch.float16
    )

    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.inverse_scheduler = DDIMInverseScheduler.from_config(pipeline.scheduler.config)
    pipeline.enable_model_cpu_offload()

    generator = torch.manual_seed(0)

    raw_image = Image.open(args.input_image).convert("RGB").resize((512,512))
    args.prompt_file=os.path.join(args.results_folder, f"prompt.txt")

    if os.path.isfile(args.prompt_file):
        caption_list = open(args.prompt_file).read().strip().split(' ')
        caption=' '.join(caption_list)

        ### NOTE: get the edit caption            
        if args.original_words is not None:
            edit_caption_list=copy.deepcopy(caption_list)
            print('replace the old word with new word')
            for replace_id in range(len(args.original_words)):
                org_word = args.original_words[replace_id]
                if org_word in caption_list: 
                    org_word_id = caption_list.index(org_word)
                else:
                    continue
                edit_caption_list[org_word_id] = args.replace_words[replace_id]
            edit_caption=' '.join(edit_caption_list)
        else:
            raise NotImplementedError
        
    source_prompts = [caption]
    target_prompts = [edit_caption]

    source_embeds = pipeline.get_embeds(source_prompts, batch_size=2)
    caption=source_prompts[0]
    inv_latents = pipeline.invert(caption, image=raw_image, generator=generator).latents
    target_embeds = pipeline.get_embeds(target_prompts, batch_size=2)

    generator = torch.manual_seed(0)

    image = pipeline(
        caption,
        source_embeds=source_embeds,
        target_embeds=target_embeds,
        num_inference_steps=50,
        cross_attention_guidance_amount=0.15,
        generator=generator,
        latents=inv_latents,
        negative_prompt=caption,
    ).images[0]

    method_name='pix2pix'
    os.makedirs(f'{args.results_folder_edit}/{method_name}/', exist_ok=True)
    image.save(f'{args.results_folder_edit}/{method_name}/{edit_caption}_edit.png')
    