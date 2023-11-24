import torch
from diffusers import DDIMScheduler
from PIL import Image

from pipelines.scheduler_inv import DDIMInverseScheduler
from pipelines.ddim_pipeline import StableDiffusionDDIMInvPipeline

import argparse
import os
import pickle as pkl
import numpy as np
from tools.sd_study_utils import *
import nltk

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=str, default=None)
    parser.add_argument('--results_folder', type=str, default=None)
    parser.add_argument('--postfix', type=str, default='')
    parser.add_argument('--num_ddim_steps', type=int, default=50)

    parser.add_argument('--model_path', type=str, default='CompVis/stable-diffusion-v1-4')
    parser.add_argument('--use_float_16', action='store_true')
    parser.add_argument('--prompt_str', type=str, default=None)
    parser.add_argument('--prompt_file', type=str, default=None)
    parser.add_argument('--search_text', type=str, default=None)

    parser.add_argument('--manual_prompt', action='store_true')
    parser.add_argument('--no-manual_prompt', dest='manual_prompt', action='store_false')
    parser.set_defaults(manual_prompt=False)

    parser.add_argument('--run_pca', action='store_true')
    parser.add_argument('--no-run_pca', dest='run_pca', action='store_false')
    parser.set_defaults(run_pca=False)

    args = parser.parse_args()
    return args


if __name__=="__main__":
    args = arguments()
    
    if args.use_float_16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    sd_model_ckpt = args.model_path

    pipeline = StableDiffusionDDIMInvPipeline.from_pretrained(
        sd_model_ckpt,
        torch_dtype=torch_dtype,
    )
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.inverse_scheduler = DDIMInverseScheduler.from_config(pipeline.scheduler.config)
    pipeline.to("cuda")

    if os.path.isfile(args.input_image):
        dirs=[args.input_image]
        dirname=os.path.dirname(args.input_image)

    elif os.path.isdir(args.input_image):
        dirs = (os.listdir(args.input_image))
        dirname=args.input_image
        dirs = [os.path.join(dirname, dir) for dir in dirs]
        dirs.sort()

    for img_path in dirs[:]:
        print(img_path)
        if os.path.isdir(args.input_image):
            search_text = img_path.split('/')[-2]
            img_id = img_path.split('/')[-1].split('.')[0]
            _results_folder = os.path.join(args.results_folder, f"{search_text}_{img_id}")
        elif os.path.isfile(args.input_image):
            img_id = img_path.split('/')[-1].split('.')[0]
            if args.search_text is None:
                _results_folder = os.path.join(args.results_folder, f"{img_id}")
            else:
                _results_folder = os.path.join(args.results_folder, f"{args.search_text}_{img_id}")

        print(_results_folder)
        os.makedirs(os.path.join(_results_folder, "org_img"), exist_ok=True)
        os.makedirs(os.path.join(_results_folder, "sd_study"), exist_ok=True)
        os.makedirs(os.path.join(_results_folder, "latents"), exist_ok=True)
        os.makedirs(os.path.join(_results_folder, "reconstruction"), exist_ok=True)
        os.makedirs(os.path.join(_results_folder, "first_recon"), exist_ok=True)
        os.makedirs(os.path.join(_results_folder, "attn_inv"), exist_ok=True)
        os.makedirs(os.path.join(_results_folder, "attn_denoise"), exist_ok=True)

        bname = img_id

        raw_image = Image.open(img_path).convert("RGB").resize((512, 512))
        raw_image.save(os.path.join(_results_folder, f"org_img/{bname}.png"))

        if args.prompt_file is None and (not args.manual_prompt):
            prompt_file=os.path.join(_results_folder, f"prompt.txt")
            caption = open(prompt_file).read().strip()
            print(f'taking caption from file {prompt_file}: \"{caption}\"')
            
        elif args.prompt_file is not None and (not args.manual_prompt):
            prompt_file=args.prompt_file
            caption = open(prompt_file).read().strip()
            print(f'taking caption from file {prompt_file}: \"{caption}\"')
        else:
            caption=args.prompt_str
            print(f'taking caption from args: \"{caption}\"  and we are {args.manual_prompt} forcing manual input ')

        generator = torch.manual_seed(0)
        all_latents, image, inv_self_avg_dict, inv_cross_avg_dict = pipeline(caption, 
                                                num_inference_steps=args.num_ddim_steps,
                                                image=raw_image, 
                                                generator=generator,
                                                )
        image[0].save(os.path.join(_results_folder, f"first_recon/{bname}.png"))

        inv_latents = all_latents
        
        image, denoise_self_avg_dict, denoise_cross_avg_dict = pipeline.reconstruct(
            caption,
            num_inference_steps=args.num_ddim_steps,
            latents=inv_latents[-1],
            guidance_scale=1.0,
            cond_embeddings_list=None,
        )

        image[0].save(os.path.join(_results_folder, f"reconstruction/{bname}.png"))
        
        with open(os.path.join(_results_folder, f"latents/{bname}.pkl"), 'wb') as f:
            pkl.dump(inv_latents, f)

        with open(os.path.join(_results_folder, 
                f"attn_inv/{bname}_inv_self_avg_dict.pkl"), 
                'wb') as f:
            pkl.dump(inv_self_avg_dict, f)
            
        with open(os.path.join(_results_folder, 
                f"attn_inv/{bname}_inv_cross_avg_dict.pkl"), 
                'wb') as f:
            pkl.dump(inv_cross_avg_dict, f)

        with open(os.path.join(_results_folder, 
                f"attn_denoise/{bname}_denoise_self_avg_dict.pkl"), 
                'wb') as f:
            pkl.dump(denoise_self_avg_dict, f)
            
        with open(os.path.join(_results_folder, 
                f"attn_denoise/{bname}_denoise_cross_avg_dict.pkl"), 
                'wb') as f:
            pkl.dump(denoise_cross_avg_dict, f)

        if args.run_pca:
            for RES in [8,16,32,64]:
                for KEY in ['feat','query','key','value', 'attn']:
                    draw_pca(denoise_self_avg_dict, resolution=RES, dict_key=KEY, 
                            save_path=os.path.join(_results_folder, 'sd_study'),
                            special_name='denoise_self')
                    
                    draw_pca(inv_self_avg_dict, resolution=RES, dict_key=KEY, 
                            save_path=os.path.join(_results_folder, 'sd_study'),
                            special_name='inv_self')
                    
                    if KEY != 'attn':
                        draw_pca(denoise_cross_avg_dict, resolution=RES, dict_key=KEY, 
                                save_path=os.path.join(_results_folder, 'sd_study'),
                                special_name='denoise_cross')
                        
                        draw_pca(inv_cross_avg_dict, resolution=RES, dict_key=KEY, 
                                save_path=os.path.join(_results_folder, 'sd_study'),
                                special_name='inv_cross')
                    
        ### NOTE: Clustering based on above
        # for RES in [8,16,32,64]:
            # for KEY in ['feat','query','key','value', 'attn']:
        for RES in [32]:
            for KEY in ['attn']:
                run_clusters(denoise_self_avg_dict, resolution=RES, dict_key=KEY, 
                        save_path=os.path.join(_results_folder, 'sd_study'),
                        special_name='denoise_self')
                
                run_clusters(inv_self_avg_dict, resolution=RES, dict_key=KEY, 
                        save_path=os.path.join(_results_folder, 'sd_study'),
                        special_name='inv_self')
                
                if KEY not in ['attn', 'key', 'value']:
                    run_clusters(denoise_cross_avg_dict, resolution=RES, dict_key=KEY, 
                            save_path=os.path.join(_results_folder, 'sd_study'),
                            special_name='denoise_cross')
                    
                    run_clusters(inv_cross_avg_dict, resolution=RES, dict_key=KEY, 
                            save_path=os.path.join(_results_folder, 'sd_study'),
                            special_name='inv_cross')
                    
        ### NOTE: save cross attentions
        # for RES in [8,16,32,64]:
        #     for KEY in ['attn']:
        #         cosine_sim = cross_cosine(RES, inv_cross_avg_dict, denoise_cross_avg_dict, 
        #                        indice=1, save_path=os.path.join(_results_folder, 'sd_study'),
        #                        save_crossattn=True,
        #                        ).cpu().item()
                
        save_crossattn(img_path, caption, inv_cross_avg_dict, denoise_cross_avg_dict, _results_folder, RES=16)

        ### NOTE: save BG.png
        dict_key='attn'
        special_name='inv_self'
        num_segments=5
        threshold=0.2
        
        tokenized_prompt = nltk.word_tokenize(caption)
        nouns = [(i, word) for (i, (word, pos)) in enumerate(nltk.pos_tag(tokenized_prompt)) if pos[:2] == 'NN']
        print(nouns)
        # for resolution in [64,32,16]:
        for resolution in [32]:
                npy_name=f'cluster_{dict_key}_{resolution}_{special_name}.npy'
                save_path=os.path.join(_results_folder, 'sd_study')

                abs_filename=os.path.join(_results_folder, "attn_inv", f"{img_id}_inv_cross_avg_dict.pkl")
                inv_cross_avg_dict=read_pkl(abs_filename)
                ### NOTE: by default we use cross attention at 16*16
                cross_attention = inv_cross_avg_dict['attn'][16]

                clusters=np.load(os.path.join(save_path, npy_name))
                c2noun, c2mask = cluster2noun_(clusters, threshold, num_segments, nouns,cross_attention)

                merged_mask={}
                for index in range(len(c2noun)):
                    item=c2noun[index]
                    mask_=c2mask[index]
                    
                    if isinstance(item, tuple):
                        label_name = item[1]
                    else:
                        label_name = item

                    if not label_name  in merged_mask.keys():
                        merged_mask[label_name]=mask_
                    else:
                        merged_mask[label_name]+=mask_

                for class_name in merged_mask.keys():
                    mask_ = merged_mask[class_name]
                    mask_ = torch.from_numpy(mask_)
                    mask_ = F.interpolate(mask_.float().unsqueeze(0).unsqueeze(0), size=512, mode='nearest').round().bool().squeeze(0).squeeze(0)
                    
                    output_name = os.path.join(f"{save_path}",
                                                f"{class_name}.png")
                                                # f"{class_name}_{dict_key}_{resolution}_cross_16_TH_{threshold}_seg_{num_segments}.png")

                    save_mask(mask_,  output_name)