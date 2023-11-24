import torch
from diffusers import DDIMScheduler
from PIL import Image
from pipelines.p2p_pipeline import StableDiffusion_EditPipeline
from pipelines.pipeline_utils import get_mapper, get_replacement_mapper_
import copy

import argparse
import os
import pickle as pkl

def arguments():
    ### NOTE: copy from zero-shot original github
    ### https://github.com/pix2pixzero/pix2pix-zero/blob/main/src/inversion.py
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=str, default=None)
    parser.add_argument('--results_folder', type=str, default='output/')
    parser.add_argument('--results_folder_edit', type=str, default='output/edit')

    parser.add_argument('--num_ddim_steps', type=int, default=50)
    parser.add_argument('--model_path', type=str, default='CompVis/stable-diffusion-v1-4')
    parser.add_argument('--use_float_16', action='store_true')

    parser.add_argument('--prompt_str', type=str, default='a cat on the grass')
    parser.add_argument('--edit_prompt_str', type=str, default='a dog on the grass')

    ### NOTE: guidance_scale  
    parser.add_argument('--negative_guidance_scale', default=7.5, type=float)
    parser.add_argument('--postfix', type=str, default='1.0_0.05_0.05_50.0_50.0_75.0_0.3_0.9_0.9_0')
    parser.add_argument('--cross_replace_steps', type=int, default=40)
    parser.add_argument('--self_replace_steps', type=int, default=40)

    ### NOTE: change this part into parameters
    parser.add_argument('--indices_to_alter', nargs='+', type=int, default=[2,])

    ### NOTE: textual inversion parameters
    parser.add_argument('--placeholder_token', nargs='+', type=str, default=['cat'])
    parser.add_argument('--initializer_token', nargs='+', type=str, default=['cat',])

    ### ATTN: 1st: p2p hyperparameters REFINE: global edit ; 
    parser.add_argument('--refine', action='store_true')
    parser.add_argument('--no-refine', dest='refine', action='store_false')
    parser.set_defaults(refine=False)

    parser.add_argument('--adj_indice', type=int, default=None)

    ### ATTN: 2nd: REPLACE: word swap
    parser.add_argument('--replace', action='store_true')
    parser.add_argument('--no-replace', dest='replace', action='store_false')
    parser.set_defaults(replace=True)
    # parser.set_defaults(replace=False)
    ### NOTE: find the original word and replace by the new word
    parser.add_argument('--original_words', nargs='+', type=str, default=['cat'])
    # parser.add_argument('--original_words', nargs='+', type=str, default=None)
    parser.add_argument('--replace_words', nargs='+', type=str, default=['dog'])
    
    ### ATTN: 3rd: local edit
    parser.add_argument('--local', action='store_true')
    parser.add_argument('--no-local', dest='local', action='store_false')
    parser.set_defaults(local=False)
    parser.add_argument('--indices_local', nargs='+', type=int, default=None)
    
    ### ATTN: 4th: reweight
    parser.add_argument('--indices_to_amplify', nargs='+', type=int, default=None)
    parser.add_argument('--amplify_scale', nargs='+', type=float, default=[1.0,1.0])

    args = parser.parse_args()
    return args


if __name__=="__main__":
    args = arguments()
    # bname = os.path.basename(args.input_image).split(".")[0]
    # os.makedirs(os.path.join(args.results_folder, "null_attend_textinv_recon"), exist_ok=True)
    os.makedirs(os.path.join(args.results_folder_edit, f"null_attend_textinv_recon_{args.postfix}"), exist_ok=True)

    # with open(os.path.join(args.results_folder, 
    #         f"embed_list_{args.postfix}/{bname}_uncond.pkl"), 
    #         'rb') as f:
    #     uncond_embeddings_list= pkl.load(f)

    # with open(os.path.join(args.results_folder, 
    #         f"embed_list_{args.postfix}/{bname}_cond.pkl"), 
    #         'rb') as f:
    #     cond_embeddings_list= pkl.load(f)

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
    
    # for ind in range(len(placeholder_token_id)):
    #     token_embeds[placeholder_token_id[ind]] = cond_embeddings_list[0][ind]
    #     index_no_updates[placeholder_token_id[ind]]=False

    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.to("cuda")

    # with open(os.path.join(args.results_folder, f"latents/{bname}.pkl"), 'rb') as f:
    #     inv_latents=pkl.load(f)
    
    ### NOTE: deal with the inversed latents
    # assert len(inv_latents)==(args.num_ddim_steps+1)
    # for ind in range(len(inv_latents)):
    #     inv_latents[ind] = inv_latents[ind].cpu().cuda()
    
    print(args)
    ### NOTE: read caption from file
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

    rec_pil = pipeline(
            caption,
            edit_caption,
            num_inference_steps=args.num_ddim_steps,
            # latents=inv_latents[-1],
            latents=None,
            guidance_scale=args.negative_guidance_scale,
            # all_latents = inv_latents,
            all_latents = None,
            placeholder_token_id=placeholder_token_id,
            index_no_updates=index_no_updates,
            token_indices = args.indices_to_alter,
            cond_embeddings_list=None,
            uncond_embeddings_list=None,
            refine=args.refine,
            replace=args.replace,
            local=args.local,
            mapper = mapper, 
            alphas = alphas,
            cross_replace_steps=args.cross_replace_steps,
            self_replace_steps=args.self_replace_steps,
            indices_to_amplify=args.indices_to_amplify,
            amplify_scale=args.amplify_scale,
            indices_local=args.indices_local,
        )
    
    file_name = '_'.join(edit_caption.split(' ') + [str(args.cross_replace_steps)])

    rec_pil[0].save(os.path.join(args.results_folder_edit, 
                    f"null_attend_textinv_recon_{args.postfix}/{file_name}.png"))
    rec_pil[1].save(os.path.join(args.results_folder_edit, 
                    f"null_attend_textinv_recon_{args.postfix}/{file_name}_edit.png"))
