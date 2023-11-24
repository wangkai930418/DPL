import torch
import PIL.Image as Image   
from diffusers import DDIMScheduler, DDIMInverseScheduler
from diffusers import StableDiffusionInstructPix2PixPipeline
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

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix", torch_dtype=torch.float16
    )
    pipe.enable_model_cpu_offload()
    # pipe = pipe.to("cuda")

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
        
    prompt=f"change the {args.original_words[0]} into {args.replace_words[0]}"
    generator = torch.manual_seed(0)
    image = pipe(prompt=prompt, 
                generator=generator,
                 image=raw_image
                 ).images[0]

    method_name='instructp2p'
    os.makedirs(f'{args.results_folder_edit}/{method_name}/', exist_ok=True)
    image.save(f'{args.results_folder_edit}/{method_name}/{edit_caption}_edit.png')
    