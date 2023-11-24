import torch
import PIL.Image as Image   
from diffusers import DDIMScheduler, DDIMInverseScheduler
from pipelines.diffedit_pipeline import StableDiffusionDiffEditPipeline
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

    pipeline = StableDiffusionDiffEditPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.inverse_scheduler = DDIMInverseScheduler.from_config(pipeline.scheduler.config)
    pipeline.enable_model_cpu_offload()
    pipeline.enable_vae_slicing()
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
        
    source_prompt = caption
    target_prompt = edit_caption

    inv_latents = pipeline.invert(prompt=source_prompt, image=raw_image, generator=generator).latents
    generator = torch.manual_seed(0)

    mask_image = pipeline.generate_mask(
        image=raw_image,
        source_prompt=source_prompt,
        target_prompt=target_prompt,
        generator=generator,
        output_type='np',
    )

    _mask_image = Image.fromarray(np.uint8(mask_image[0])*255)

    os.makedirs(f'{args.results_folder_edit}/diffedit/', exist_ok=True)
    _mask_image.save(f'{args.results_folder_edit}/diffedit/{edit_caption}_mask.png')

    image = pipeline(
        prompt=target_prompt,
        mask_image=mask_image,
        image_latents=inv_latents,
        generator=generator,
        negative_prompt=source_prompt,
    ).images[0]

    image.save(f'{args.results_folder_edit}/diffedit/{edit_caption}_edit.png')
    