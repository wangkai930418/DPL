import torch
from diffusers import DDIMScheduler
from PIL import Image
from pipelines.seg_null_textinv_pipeline import StableDiffusion_SegPipeline
from _utils.ptp_utils import show_cross_attention, show_cross_attention_plus_orig_img,show_cross_attention_blackwhite, save_attn_avg, mean_iou

import argparse
import os
import pickle as pkl
import numpy as np

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
    parser.add_argument('--lam_cos', default=1.0, type=float)
    parser.add_argument('--lam_iou', default=1.0, type=float)
    parser.add_argument('--lam_kl', default=1.0, type=float)
    parser.add_argument('--lam_sim', default=0.0, type=float)
    parser.add_argument('--lam_adj', default=2.0, type=float)

    ###NOTE: alpha_cos >= 50.0
    parser.add_argument('--alpha_cos', default=50.0 , type=float)
    parser.add_argument('--alpha_iou', default=25.0, type=float)
    parser.add_argument('--alpha_kl', default=25.0 , type=float)
    parser.add_argument('--alpha_sim', default=25.0 , type=float)
    parser.add_argument('--alpha_adj', default=50.0 , type=float)
    ### NOTE: beta_cos >=0.6
    parser.add_argument('--beta_cos', default=0.7, type=float)
    parser.add_argument('--beta_iou', default=0.7, type=float)
    parser.add_argument('--beta_kl', default=1.0, type=float)
    parser.add_argument('--beta_sim', default=0.9, type=float)
    parser.add_argument('--beta_adj', default=0.1, type=float)
    # parser.add_argument('--beta_adj', default=0.01, type=float)

    ### NOTE: cosine iou
    parser.add_argument('--loss_type', type=str, default='cosine')

    parser.add_argument('--null_inner_steps', type=int, default=31)
    parser.add_argument('--attn_inner_steps', type=int, default=31)  
    
    parser.add_argument('--max_iter_to_alter', type=int, default=0)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--indices_to_alter', nargs='+', type=int, default=None)

    parser.add_argument('--attn_res', type=int, default=16)

    ### NOTE: the next two operations will change the real attention maps
    parser.add_argument('--smooth_op', action='store_true')
    parser.add_argument('--no-smooth_op', dest='smooth_op', action='store_false')
    parser.set_defaults(smooth_op=True)

    parser.add_argument('--softmax_op', action='store_true')
    parser.add_argument('--no-softmax_op', dest='softmax_op', action='store_false')
    parser.set_defaults(softmax_op=True)

    parser.add_argument('--adj_bind', action='store_true')
    parser.add_argument('--no-adj_bind', dest='adj_bind', action='store_false')
    parser.set_defaults(adj_bind=False)

    ### NOTE: textual inversion parameters
    parser.add_argument('--placeholder_token', nargs='+', type=str, default=None)
    parser.add_argument('--initializer_token', nargs='+', type=str, default=None)
    args = parser.parse_args()
    return args


if __name__=="__main__":
    args = arguments()
    print(args)
    torch_dtype = torch.float32
    sd_model_ckpt = args.model_path
    
    ### NOTE: change the postfix later
    postfix = f'cos_al_{args.alpha_cos}_beta_{args.beta_cos}_lam_{args.lam_cos}_'+ \
                f'iou_al_{args.alpha_iou}_beta_{args.beta_iou}_lam_{args.lam_iou}_'+ \
                    f'kl_al_{args.alpha_kl}_beta_{args.beta_kl}_lam_{args.lam_kl}_'+ \
                        f'adj_al_{args.alpha_adj}_beta_{args.beta_adj}_lam_{args.lam_adj}_'+ \
                            f'softmax_{args.softmax_op}_smooth_{args.smooth_op}'+ \
                                f'_null_{args.null_inner_steps}_attn_{args.attn_inner_steps}'+ \
                                    f'_CFG_{args.negative_guidance_scale}_adj_{args.adj_bind}'

    print(postfix)
    
    os.makedirs(os.path.join(args.results_folder, 
                             f"attn/{postfix}"), 
                             exist_ok=True)
    os.makedirs(os.path.join(args.results_folder, 
                             f"null_inv_recon/{postfix}"), 
                             exist_ok=True)
    os.makedirs(os.path.join(args.results_folder, 
                             f"embed_list/{postfix}"), 
                             exist_ok=True)
    
    pipeline = StableDiffusion_SegPipeline.from_pretrained(
        sd_model_ckpt,
        torch_dtype=torch_dtype,
    )
    ### NOTE: for textual inversion https://huggingface.co/docs/diffusers/training/text_inversion
    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder
    num_added_tokens = tokenizer.add_tokens(args.placeholder_token)
    token_ids = tokenizer.encode(args.initializer_token, add_special_tokens=False)

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
        seg_search_words=[]

        for ind in range(len(placeholder_token_id)):
            if args.initializer_token[ind] in caption_list:
                plh_id = caption_list.index(args.initializer_token[ind])
                seg_search_words.append(args.initializer_token[ind])
            else:
                continue
            caption_list[plh_id] = args.placeholder_token[ind]
            ### NOTE: change this part to create a new list
            args.indices_to_alter.append(plh_id+1) 
            caption=' '.join(caption_list)
        print(f'taking caption from file: \"{caption}\"')
        print(f'alter the indices: {args.indices_to_alter}')
    else:
        caption_=args.prompt_str
        caption_list = caption_.strip().split(' ')
        args.indices_to_alter=[]
        seg_search_words=[]

        for ind in range(len(placeholder_token_id)):
            if args.initializer_token[ind] in caption_list:
                plh_id = caption_list.index(args.initializer_token[ind])
                seg_search_words.append(args.initializer_token[ind])
            else:
                continue
            caption_list[plh_id] = args.placeholder_token[ind]
            ### NOTE: change this part to create a new list
            args.indices_to_alter.append(plh_id+1) 
            caption=' '.join(caption_list)

        print(f'taking caption from args: \"{caption}\"')
        print(f'alter the indices: {args.indices_to_alter}')

    ######## ================================================
    ### NOTE: read segmentation maps
    seg_maps=[]
    seg_maps_paths=[]
    for ind in range(len(seg_search_words)):
        object_name = seg_search_words[ind]

        if os.path.isfile(os.path.join(args.seg_dirs, f"{object_name}.png")):
            seg_image_path = os.path.join(args.seg_dirs, f"{object_name}.png")
        else:
            seg_image_path = os.path.join(args.seg_dirs, f"{object_name}.jpg")

        seg_maps_paths.append(seg_image_path)

        print(f'read segmentation map from {seg_image_path}')
        seg_maps.append(read_segfile(seg_image_path))
    ######## ================================================

    if args.adj_bind:
        adj_indices_to_alter = [x-1 for x in args.indices_to_alter]
    else:
        adj_indices_to_alter=None
        
    rec_pil_train, attention_maps, uncond_embeddings_list, cond_embeddings_list = pipeline(
        caption,
        num_inference_steps=args.num_ddim_steps,
        latents=inv_latents[-1],
        guidance_scale=args.negative_guidance_scale,
        all_latents = inv_latents,
        print_freq=args.print_freq,
        null_inner_steps=args.null_inner_steps,
        attn_inner_steps=args.attn_inner_steps,
        placeholder_token_id=placeholder_token_id,
        index_no_updates=index_no_updates,
        token_indices = args.indices_to_alter,
        adj_indices_to_alter=adj_indices_to_alter,
        alpha_cos = args.alpha_cos,
        alpha_iou = args.alpha_iou,
        alpha_kl = args.alpha_kl,
        alpha_sim = args.alpha_sim,
        alpha_adj = args.alpha_adj,
        beta_cos = args.beta_cos,
        beta_iou = args.beta_iou,
        beta_kl = args.beta_kl,
        beta_sim = args.beta_sim,
        beta_adj = args.beta_adj,
        lam_cos = args.lam_cos,
        lam_iou = args.lam_iou,
        lam_kl = args.lam_kl,
        lam_sim = args.lam_sim,
        lam_adj = args.lam_adj,
        attn_res=args.attn_res,
        smooth_op=args.smooth_op,
        softmax_op = args.softmax_op,
        seg_maps=seg_maps,
        loss_type=args.loss_type,
    )

    with open(os.path.join(args.results_folder, 
            f"embed_list/{postfix}/{bname}_uncond.pkl"), 
            'wb') as f:
        pkl.dump(uncond_embeddings_list, f)

    with open(os.path.join(args.results_folder, 
            f"embed_list/{postfix}/{bname}_cond.pkl"), 
            'wb') as f:
        pkl.dump(cond_embeddings_list, f)

    rec_pil = pipeline.reconstruct(
            caption,
            num_inference_steps=args.num_ddim_steps,
            latents=inv_latents[-1],
            guidance_scale=args.negative_guidance_scale,
            placeholder_token_id=placeholder_token_id,
            index_no_updates=index_no_updates,
            token_indices = args.indices_to_alter,
            cond_embeddings_list=cond_embeddings_list,
            uncond_embeddings_list=uncond_embeddings_list,
        )

    rec_pil[0].save(os.path.join(args.results_folder, 
            f"null_inv_recon/{postfix}/{bname}.png"))
    
    with open(os.path.join(args.results_folder, 
            f"attn/{postfix}/{bname}.pkl"), 
            'wb') as f:
        pkl.dump(attention_maps, f)

    ### NOTE: save the averaged attention map in figures
    prompts_ = ["<|startoftext|>",] + caption_list + ["<|endoftext|>",]

    attn_maps = [item.unsqueeze(0) for item in attention_maps]
    attn_maps = torch.cat(attn_maps).mean(dim=0)
    print(attn_maps.shape)

    single_attn_paths = save_attn_avg(save_path=os.path.join(args.results_folder, f"attn/{postfix}"), 
                img_path=args.input_image, 
                caption_list=prompts_, 
                aggr_attn=attn_maps, 
                placeholders_list=args.placeholder_token, 
                show_orig_img=True, 
                image_size=512,)
    
    ### NOTE: compute and save iou
    IoU_txt_file = os.path.join(args.results_folder, f"attn/{postfix}", 'iou.txt')
    IoUs=[]
    for object_id in range(len(seg_maps_paths)):
        IoU = mean_iou(seg_maps_paths[object_id], single_attn_paths[object_id], Threshold=0.3)
        print(IoU)
        IoUs.append(str(IoU))

    with open(IoU_txt_file, "w") as f:
            f.write(','.join(IoUs))