import torch
from diffusers import DDIMScheduler
from PIL import Image
from pipelines.null_attend_textinv_pipeline import StableDiffusion_MyPipeline

import argparse
import os
import pickle as pkl
import numpy as np
from _utils.ptp_utils import *

import warnings
warnings.filterwarnings("ignore")

def read_segfile(seg_image_path):
    seg_image = Image.open(seg_image_path).resize((16,16))
    seg_img_data = np.asarray(seg_image).astype(bool)
    if len(seg_img_data.shape) >2:
        seg_img_data=seg_img_data[:,:,-1]
    seg_img_data = torch.from_numpy(seg_img_data).to(torch.float32).cuda().unsqueeze(0)
    return seg_img_data


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=str, default=None)
    parser.add_argument('--results_folder', type=str, default=None)
    parser.add_argument('--seg_dirs', type=str, default=None)

    parser.add_argument('--num_ddim_steps', type=int, default=50)
    parser.add_argument('--model_path', type=str, default='CompVis/stable-diffusion-v1-4')
    parser.add_argument('--use_float_16', action='store_true')
    parser.add_argument('--prompt_str', type=str, default=None)
    parser.add_argument('--prompt_file', type=str, default=None)

    ### NOTE: guidance_scale
    parser.add_argument('--negative_guidance_scale', default=7.5, type=float)

    ### NOTE: change this part into parameters
    parser.add_argument('--lam_maxattn', default=0.0, type=float)
    parser.add_argument('--lam_entropy', default=0.0, type=float)
    parser.add_argument('--lam_cosine', default=0.0, type=float)

    ### NOTE: exp(-epoch/alpha)*beta
    parser.add_argument('--alpha_max', default=25.0, type=float)
    parser.add_argument('--alpha_ent', default=50.0 , type=float)
    parser.add_argument('--alpha_cos', default=25.0 , type=float)

    parser.add_argument('--beta_max', default=0.3, type=float)
    parser.add_argument('--beta_ent', default=0.3, type=float)
    parser.add_argument('--beta_cos', default=0.9, type=float)
    
    parser.add_argument('--loss_type', type=str, default='max')
    ### NOTE: mean is not useful. 'max' is a better choice

    parser.add_argument('--null_inner_steps', type=int, default=51)
    parser.add_argument('--attn_inner_steps', type=int, default=1)  
    parser.add_argument('--max_iter_to_alter', type=int, default=0)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--indices_to_alter', nargs='+', type=int, default=None)

    parser.add_argument('--attn_res', type=int, default=16)

    parser.add_argument('--smooth_op', action='store_true')
    parser.add_argument('--no-smooth_op', dest='smooth_op', action='store_false')
    parser.set_defaults(smooth_op=True)

    parser.add_argument('--softmax_op', action='store_true')
    parser.add_argument('--no-softmax_op', dest='softmax_op', action='store_false')
    parser.set_defaults(softmax_op=True)

    ### NOTE: textual inversion parameters
    parser.add_argument('--placeholder_token', nargs='+', type=str, default=None)
    parser.add_argument('--initializer_token', nargs='+', type=str, default=None)
    args = parser.parse_args()
    return args


if __name__=="__main__":
    args = arguments()
    torch_dtype = torch.float32
    sd_model_ckpt = args.model_path
    
    postfix = f'{args.lam_maxattn}_{args.lam_entropy}_{args.lam_cosine}' + \
                f'_{args.alpha_max}_{args.alpha_ent}_{args.alpha_cos}' + \
                    f'_{args.beta_max}_{args.beta_ent}_{args.beta_cos}_{args.max_iter_to_alter}'

    os.makedirs(os.path.join(args.results_folder, 
                             f"attn_{postfix}"), 
                             exist_ok=True)
    os.makedirs(os.path.join(args.results_folder, 
                             f"null_inv_recon_{postfix}"), 
                             exist_ok=True)
    os.makedirs(os.path.join(args.results_folder, 
                             f"embed_list_{postfix}"), 
                             exist_ok=True)
    
    pipeline = StableDiffusion_MyPipeline.from_pretrained(
        sd_model_ckpt,
        torch_dtype=torch_dtype,
    )
    ### NOTE: ===============================
    ### for textual inversion https://huggingface.co/docs/diffusers/training/text_inversion
    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder
    num_added_tokens = tokenizer.add_tokens(args.placeholder_token)
    token_ids = tokenizer.encode(args.initializer_token, add_special_tokens=False)
    # initializer_token_id = token_ids[0]
    initializer_token_id = token_ids
    placeholder_token_id = tokenizer.convert_tokens_to_ids(args.placeholder_token)    

    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    text_encoder.resize_token_embeddings(len(tokenizer))

    # Initialise the newly added placeholder token with the embeddings of the initializer token
    token_embeds = text_encoder.get_input_embeddings().weight.data
    
    index_no_updates = torch.ones(len(tokenizer), dtype=bool)
    
    for ind in range(len(placeholder_token_id)):
        token_embeds[placeholder_token_id[ind]] = token_embeds[initializer_token_id[ind]]
        index_no_updates[placeholder_token_id[ind]]=False
        
    # NOTE: Freeze all parameters except for the token embeddings in text encoder
    text_encoder.text_model.encoder.requires_grad_(False)
    text_encoder.text_model.final_layer_norm.requires_grad_(False)
    text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)

    ### NOTE: https://github.com/huggingface/diffusers/tree/main/examples/textual_inversion
    ### =============================================================================================

    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.to("cuda")

    bname = os.path.basename(args.input_image).split(".")[0]

    with open(os.path.join(args.results_folder, f"latents/{bname}.pkl"), 'rb') as f:
        inv_latents=pkl.load(f)
    
    ### NOTE: deal with the inversed latents
    assert len(inv_latents)==(args.num_ddim_steps+1)
    for ind in range(len(inv_latents)):
        inv_latents[ind] = inv_latents[ind].cpu().cuda()
    
    ### NOTE: include prompt files if provides
    if args.prompt_file is None:
        args.prompt_file=os.path.join(args.results_folder, f"prompt.txt")
        
    if os.path.isfile(args.prompt_file):
        caption_list = open(args.prompt_file).read().strip().split(' ')
        args.indices_to_alter=[]
        for ind in range(len(placeholder_token_id)):
            if args.initializer_token[ind] in caption_list:
                plh_id = caption_list.index(args.initializer_token[ind])
            else:
                continue
            caption_list[plh_id] = args.placeholder_token[ind]
            ### NOTE: change this part to create a new list
            args.indices_to_alter.append(plh_id+1) 
            caption=' '.join(caption_list)
        print(f'taking caption from file: \"{caption}\"')
        print(f'alter the indices: {args.indices_to_alter}')
    else:
        caption=args.prompt_str
        print(f'taking caption from args: \"{caption}\"')
        print(f'alter the indices: {args.indices_to_alter}')
        
    ######## ================================================
    ### NOTE: read segmentation maps
    # BG_maps=[]
    BG_image_path = os.path.join(args.seg_dirs, f"BG.png")

    if os.path.isfile(BG_image_path):
        print(f'read background map from {BG_image_path}')
        BG_maps = read_segfile(BG_image_path)
    else:
        BG_maps=None
        print('no BG maps offered')
    ######## ================================================

    rec_pil, attention_maps, uncond_embeddings_list, cond_embeddings_list = pipeline(
        caption,
        num_inference_steps=args.num_ddim_steps,
        latents=inv_latents[-1],
        guidance_scale=args.negative_guidance_scale,
        all_latents = inv_latents,
        print_freq=args.print_freq,
        null_inner_steps=args.null_inner_steps,
        attn_inner_steps=args.attn_inner_steps,
        placeholder_token_id=placeholder_token_id,
        lam_maxattn=args.lam_maxattn,
        lam_entropy=args.lam_entropy,
        lam_cosine=args.lam_cosine,
        index_no_updates=index_no_updates,
        token_indices = args.indices_to_alter,
        max_iter_to_alter = args.max_iter_to_alter,
        alpha_max = args.alpha_max,
        alpha_ent = args.alpha_ent,
        alpha_cos = args.alpha_cos,
        beta_max = args.beta_max,
        beta_ent = args.beta_ent,
        beta_cos = args.beta_cos,
        loss_type = args.loss_type, 
        attn_res=args.attn_res,
        smooth_op=args.smooth_op,
        softmax_op = args.softmax_op,
        BG_maps=BG_maps,
    )

    with open(os.path.join(args.results_folder, 
            f"embed_list_{postfix}/{bname}_uncond.pkl"), 
            'wb') as f:
        pkl.dump(uncond_embeddings_list, f)

    with open(os.path.join(args.results_folder, 
            f"embed_list_{postfix}/{bname}_cond.pkl"), 
            'wb') as f:
        pkl.dump(cond_embeddings_list, f)

    rec_pil[0].save(os.path.join(args.results_folder, 
            f"null_inv_recon_{postfix}/{bname}.png"))
    
    ### save the cross-attention maps plus the original image
    if len(attention_maps)>0:
        with open(os.path.join(args.results_folder, 
                f"attn_{postfix}/{bname}.pkl"), 
                'wb') as f:
            pkl.dump(attention_maps, f)

        org_image = Image.open(args.input_image)
        prompts=["<|startoftext|>",] + caption.split(' ') + ["<|endoftext|>",]
        attn_maps = [item.unsqueeze(0) for item in attention_maps]
        attn_maps = torch.cat(attn_maps).mean(dim=0)
        attn_img, _ = show_cross_attention_plus_orig_img(prompts, attn_maps, orig_image=org_image)
        attn_img.save(os.path.join(args.results_folder,f'crossattn_ours_{postfix}.png'))