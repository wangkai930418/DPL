import torch
import PIL.Image as Image   
from diffusers import DDIMScheduler, DDIMInverseScheduler
from pipelines.imagic_pipeline import ImagicStableDiffusionPipeline
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
    
    pipe = ImagicStableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
            safety_checker=None,
        use_auth_token=True,
        custom_pipeline="imagic_stable_diffusion",
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    )
    pipe.to("cuda")

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
        
    res = pipe.train(
        edit_caption,
        image=raw_image,
        generator=generator,
        text_embedding_optimization_steps=100,
        model_fine_tuning_optimization_steps=100,)

        
    res = pipe(alpha=1, guidance_scale=7.5, num_inference_steps=50)
    image = res.images[0]
    
    method_name='imagic'
    os.makedirs(f'{args.results_folder_edit}/{method_name}/', exist_ok=True)
    image.save(f'{args.results_folder_edit}/{method_name}/{edit_caption}_edit.png')
    