import torch
from diffusers import DDIMScheduler
from PIL import Image
from pipelines.edit_pipeline import StableDiffusion_EditPipeline

from pipelines.pipeline_utils import get_mapper, get_replacement_mapper_
import copy
import numpy as np
import argparse
import os
import pickle as pkl

def read_segfile(seg_image_path, res=16):
    seg_image = Image.open(seg_image_path).resize((res,res))
    seg_img_data = np.asarray(seg_image).astype(bool)
    if len(seg_img_data.shape) >2:
        seg_img_data=seg_img_data[:,:,-1]
    seg_img_data = torch.from_numpy(seg_img_data).to(torch.float32).cuda().unsqueeze(0)
    return seg_img_data

def arguments():
    ### NOTE: copy from zero-shot original github
    ### https://github.com/pix2pixzero/pix2pix-zero/blob/main/src/inversion.py  
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=str, default=None)
    parser.add_argument('--results_folder', type=str, default=None)
    parser.add_argument('--results_folder_edit', type=str, default=None)
    parser.add_argument('--seg_dirs', type=str, default=None)

    parser.add_argument('--num_ddim_steps', type=int, default=50)
    parser.add_argument('--model_path', type=str, default='CompVis/stable-diffusion-v1-4')
    parser.add_argument('--use_float_16', action='store_true')

    parser.add_argument('--prompt_str', type=str, default=None)
    parser.add_argument('--edit_prompt_str', type=str, default=None)

    ### NOTE: CFG: guidance_scale. Bigger guidance may lead to wrong reconstructions 
    parser.add_argument('--negative_guidance_scale', default=7.5, type=float)

    parser.add_argument('--cross_replace_steps', type=int, default=50)
    parser.add_argument('--self_replace_steps', type=int, default=50)
    parser.add_argument('--self_start_steps', type=int, default=-1)

    ### NOTE: change this part into parameters
    parser.add_argument('--lam_cos', default=0.5, type=float)
    parser.add_argument('--lam_iou', default=1.0, type=float)
    parser.add_argument('--lam_kl', default=0.0, type=float)
    parser.add_argument('--lam_sim', default=0.0, type=float)
    parser.add_argument('--lam_adj', default=0.0, type=float)
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

    parser.add_argument('--null_inner_steps', type=int, default=31)
    parser.add_argument('--attn_inner_steps', type=int, default=31)  

    ### NOTE: the next two operations will change the real attention maps
    parser.add_argument('--smooth_op', action='store_true')
    parser.add_argument('--no-smooth_op', dest='smooth_op', action='store_false')
    parser.set_defaults(smooth_op=True)
    parser.add_argument('--softmax_op', action='store_true')
    parser.add_argument('--no-softmax_op', dest='softmax_op', action='store_false')
    parser.set_defaults(softmax_op=True)

    ### NOTE: change this part into parameters
    parser.add_argument('--indices_to_alter', nargs='+', type=int, default=None)

    ### NOTE: textual inversion parameters
    parser.add_argument('--placeholder_token', nargs='+', type=str, default=None)
    parser.add_argument('--initializer_token', nargs='+', type=str, default=None)

    ### ATTN: 1st: p2p hyperparameters REFINE: global edit; 
    parser.add_argument('--refine', action='store_true')
    parser.add_argument('--no-refine', dest='refine', action='store_false')
    parser.set_defaults(refine=False)

    parser.add_argument('--adj_indice', type=int, default=None)

    ### ATTN: 2nd: REPLACE: word swap
    parser.add_argument('--replace', action='store_true')
    parser.add_argument('--no-replace', dest='replace', action='store_false')
    parser.set_defaults(replace=True)
    # parser.set_defaults(replace=False)
    # ### NOTE: find the original word and replace by the new word
    parser.add_argument('--original_words', nargs='+', type=str, default=None)
    parser.add_argument('--replace_words', nargs='+', type=str, default=None)

    ### ATTN: 3rd: local edit
    parser.add_argument('--local', action='store_true')
    parser.add_argument('--no-local', dest='local', action='store_false')
    parser.set_defaults(local=False)
    # parser.set_defaults(local=True)

    ### NOTE: not used now CA_mask
    parser.add_argument('--CA_mask', action='store_true')
    parser.add_argument('--no-CA_mask', dest='CA_mask', action='store_false')
    parser.set_defaults(CA_mask=False)
    # parser.set_defaults(CA_mask=True)

    parser.add_argument('--SA_blend', action='store_true')
    parser.add_argument('--no-SA_blend', dest='SA_blend', action='store_false')
    parser.set_defaults(SA_blend=False)
    # parser.set_defaults(SA_blend=True)

    parser.add_argument('--indices_local', nargs='+', type=int, default=None)
    
    ### ATTN: 4th: reweight
    parser.add_argument('--indices_to_amplify', nargs='+', type=int, default=None)
    parser.add_argument('--amplify_scale', nargs='+', type=float, default=None)

    parser.add_argument('--adj_bind', action='store_true')
    parser.add_argument('--no-adj_bind', dest='adj_bind', action='store_false')
    parser.set_defaults(adj_bind=False)

    args = parser.parse_args()
    return args


if __name__=="__main__":
    args = arguments()
    bname = os.path.basename(args.input_image).split(".")[0]

    postfix = f'cos_al_{args.alpha_cos}_beta_{args.beta_cos}_lam_{args.lam_cos}_'+ \
                f'iou_al_{args.alpha_iou}_beta_{args.beta_iou}_lam_{args.lam_iou}_'+ \
                    f'kl_al_{args.alpha_kl}_beta_{args.beta_kl}_lam_{args.lam_kl}_'+ \
                        f'adj_al_{args.alpha_adj}_beta_{args.beta_adj}_lam_{args.lam_adj}_'+ \
                            f'softmax_{args.softmax_op}_smooth_{args.smooth_op}'+ \
                                f'_null_{args.null_inner_steps}_attn_{args.attn_inner_steps}'+ \
                                    f'_CFG_{args.negative_guidance_scale}_adj_{args.adj_bind}'
    
    os.makedirs(os.path.join(args.results_folder_edit, f"{postfix}"), exist_ok=True)

    with open(os.path.join(args.results_folder, 
            f"embed_list/{postfix}/{bname}_uncond.pkl"), 
            'rb') as f:
        uncond_embeddings_list= pkl.load(f)

    with open(os.path.join(args.results_folder, 
            f"embed_list/{postfix}/{bname}_cond.pkl"), 
            'rb') as f:
        cond_embeddings_list= pkl.load(f)

    torch_dtype = torch.float32
    sd_model_ckpt = args.model_path
    
    pipeline = StableDiffusion_EditPipeline.from_pretrained(
        sd_model_ckpt,
        torch_dtype=torch_dtype,
    )

    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder
    num_added_tokens = tokenizer.add_tokens(args.placeholder_token)
    token_ids = tokenizer.encode(args.initializer_token, add_special_tokens=False)
    initializer_token_id = token_ids
    placeholder_token_id = tokenizer.convert_tokens_to_ids(args.placeholder_token)    
    text_encoder.resize_token_embeddings(len(tokenizer))
    token_embeds = text_encoder.get_input_embeddings().weight.data
    
    index_no_updates = torch.ones(len(tokenizer), dtype=bool)
    
    for ind in range(len(placeholder_token_id)):
        token_embeds[placeholder_token_id[ind]] = cond_embeddings_list[0][ind]
        index_no_updates[placeholder_token_id[ind]]=False

    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.to("cuda")

    with open(os.path.join(args.results_folder, f"latents/{bname}.pkl"), 'rb') as f:
        inv_latents=pkl.load(f)
    
    ### NOTE: deal with the inversed latents
    assert len(inv_latents)==(args.num_ddim_steps+1)
    for ind in range(len(inv_latents)):
        inv_latents[ind] = inv_latents[ind].cpu().cuda()
    
    print(args)
    ### NOTE: read caption from file
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
            args.indices_to_alter.append(plh_id+1) 
            caption=' '.join(caption_list)

        ### NOTE: get the edit caption            
        if args.replace and (args.original_words is not None):
            edit_caption_list=copy.deepcopy(caption_list)
            print('replace the old word with new word')
            for replace_id in range(len(args.original_words)):
                org_word = args.original_words[replace_id]
                if org_word in caption_list: 
                    org_word_id = caption_list.index(org_word)

                    ### NOTE:
                    if org_word in args.placeholder_token:
                        seg_map_id = args.placeholder_token.index(org_word)
                    else:
                        ###HARDCODE: by default the adjective is before the noun
                        seg_map_id = args.placeholder_token.index(caption_list[org_word_id+1])

                else:
                    continue
                edit_caption_list[org_word_id] = args.replace_words[replace_id]
            edit_caption=' '.join(edit_caption_list)
        else:
            edit_caption=args.edit_prompt_str
            if edit_caption is None:
                edit_caption = caption
        # breakpoint()
        print(f'taking caption from file: \"{caption}\"')
        print(f'taking edit_caption from file: \"{edit_caption}\"')
        print(f'alter the indices: {args.indices_to_alter}')
        print(f'segmentation map id is: {seg_map_id}')
    else:
        caption=args.prompt_str
        edit_caption=args.edit_prompt_str
        print(f'taking caption from args: \"{caption}\"')
        print(f'taking edit_caption from args: \"{edit_caption}\"')
        print(f'alter the indices: {args.indices_to_alter}')
    
    if args.refine:
        mapper, alphas = get_mapper(caption, edit_caption, tokenizer)
        mapper, alphas = mapper.cuda(), alphas.cuda()
    else:
        mapper, alphas = None, None

    if args.adj_indice is not None:
        mapper[args.adj_indice] = args.adj_indice
        alphas[args.adj_indice] = 1.0

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
        seg_maps.append(read_segfile(seg_image_path, res=64))
    ######## ================================================

    rec_pil = pipeline(
            caption,
            edit_caption,
            num_inference_steps=args.num_ddim_steps,
            latents=inv_latents[-1],
            guidance_scale=args.negative_guidance_scale,
            all_latents = inv_latents,
            placeholder_token_id=placeholder_token_id,
            index_no_updates=index_no_updates,
            token_indices = args.indices_to_alter,
            cond_embeddings_list=cond_embeddings_list,
            uncond_embeddings_list=uncond_embeddings_list,
            refine=args.refine,
            replace=args.replace,
            local=args.local,
            CA_mask=args.CA_mask,
            mapper = mapper, 
            alphas = alphas,
            cross_replace_steps=args.cross_replace_steps,
            self_replace_steps=args.self_replace_steps,
            self_start_steps=args.self_start_steps,
            indices_to_amplify=args.indices_to_amplify,
            amplify_scale=args.amplify_scale,
            indices_local=args.indices_local,
            seg_maps=seg_maps,
            seg_map_id=seg_map_id,
            SA_blend=args.SA_blend,
        )
    
    file_name = '_'.join(edit_caption.split(' ') + [f"cross_{args.cross_replace_steps}",f"self_{args.self_replace_steps}",f"blend_{args.SA_blend}"])

    rec_pil[0].save(os.path.join(args.results_folder_edit, 
                    f"{postfix}/{file_name}.png"))
    rec_pil[1].save(os.path.join(args.results_folder_edit, 
                    f"{postfix}/{file_name}_edit.png"))