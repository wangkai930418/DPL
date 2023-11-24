import math
from typing import List, Tuple, Union
from PIL import Image
import cv2
import numpy as np
import torch
from IPython.display import display
import os

def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    img[:h] = image
    textsize = cv2.getTextSize(text, font, fontScale=1, thickness=2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y), font, 1, text_color, 2)
    return img

def show_cross_attention_blackwhite(prompts,
                         attention_maps, 
                         display_image=True,
                         ):
    # tokens = tokenizer.encode(prompts[select])
    # decoder = tokenizer.decode
    # attention_maps = aggregate_attention(attention_store, res, from_where, True, select)
    images = []
    split_imgs = []
    for i in range(len(prompts)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.cpu().numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        split_imgs.append(image)
        
        image = text_under_image(image, prompts[i])
        images.append(image)
    pil_img=view_images(np.stack(images, axis=0),display_image=display_image)
    return pil_img, split_imgs

def show_cross_attention(prompts,
                         attention_maps, 
                         display_image=True,
                         ):
    # tokens = tokenizer.encode(prompts[select])
    # decoder = tokenizer.decode
    # attention_maps = aggregate_attention(attention_store, res, from_where, True, select)
    images = []
    split_imgs = []
    white_image = Image.new('RGB', (500, 500), (255, 255, 255))
    
    for i in range(len(prompts)):
        image = attention_maps[:, :, i]
        image = show_image_relevance(image, white_image)
        
        image = image.astype(np.uint8)
        
        # image = 255 * image / image.max()
        # image = image.unsqueeze(-1).expand(*image.shape, 3)
        # image = image.numpy().astype(np.uint8)
        
        image = np.array(Image.fromarray(image).resize((256, 256)))
        split_imgs.append(image)
        # image = text_under_image(image, decoder(int(tokens[i])))
        image = text_under_image(image, prompts[i])
        images.append(image)
    pil_img=view_images(np.stack(images, axis=0),display_image=display_image)
    return pil_img, split_imgs


def show_cross_attention_plus_orig_img(
                        prompts,
                        attention_maps, 
                        display_image=True,
                        orig_image=None,
                        indices_to_alter=None,
                        res=16,
                        ):
    images = []
    split_imgs = []
    if indices_to_alter is None:
        indices_to_alter=list(range(len(prompts)))
    for i in range(len(prompts)):
        image = attention_maps[:, :, i]
        if i in indices_to_alter:
            image = show_image_relevance(image, orig_image)
            image = image.astype(np.uint8)
            image = np.array(Image.fromarray(image).resize((res ** 2, res ** 2)))
            split_imgs.append(image)

            image = text_under_image(image, prompts[i])
            images.append(image)

    pil_img=view_images(np.stack(images, axis=0),display_image=display_image)
    return pil_img, images


def show_image_relevance(image_relevance, image: Image.Image, relevnace_res=16):
    # create heatmap from mask on image
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam

    image = image.resize((relevnace_res ** 2, relevnace_res ** 2))
    image = np.array(image)

    image_relevance = image_relevance.reshape(1, 1, image_relevance.shape[-1], image_relevance.shape[-1])
    image_relevance = image_relevance.cuda() # because float16 precision interpolation is not supported on cpu
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=relevnace_res ** 2, mode='bilinear')
    image_relevance = image_relevance.cpu() # send it back to cpu
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    image_relevance = image_relevance.reshape(relevnace_res ** 2, relevnace_res ** 2)
    image = (image - image.min()) / (image.max() - image.min()+1e-8)
    vis = show_cam_on_image(image, image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis

def view_images(images: Union[np.ndarray, List],
                num_rows: int = 1,
                offset_ratio: float = 0.02,
                display_image: bool = True) -> Image.Image:
    """ Displays a list of images in a grid. """
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    if display_image:
        display(pil_img)
    return pil_img



def save_attn_avg(save_path, img_path, caption_list, aggr_attn, placeholders_list, show_orig_img=True, image_size=512,):
    # for file in files:
        # img_path=os.path.join(args.input_dir, dir,file)
        # img_id=file.split('.')[0]

        # FILE_NAME=dir+'_'+img_id
        # prompt_file=os.path.join('output', FILE_NAME, f"prompt.txt")
        # print(prompt_file)
        # caption_list = open(prompt_file).read().strip().split(' ')
        # caption_list=["<|startoftext|>",] + caption_list + ["<|endoftext|>",]

        keyword_ind=[]
        # for word_id in range(len(caption_list)):
        #     if caption_list[word_id] in placeholders_list:
        #         # caption_list[word_id]= '<' + caption_list[word_id]+'>'
        #         keyword_ind.append((word_id,caption_list[word_id]))

        for placeholder in placeholders_list:
            if placeholder in caption_list:
                word_id = caption_list.index(placeholder)
                keyword_ind.append((word_id,placeholder))
                
        # attn_file = f"output/{FILE_NAME}/{args.attn_path}/{img_id}.pkl"
        # if not os.path.isfile(attn_file):
        #     continue
        # with open(attn_file, 'rb') as f:
        #     attention_maps=pkl.load(f)

        # for attn_id in range(len(attention_maps)):
        #     attention_maps[attn_id]=attention_maps[attn_id].reshape(-1,16,16,77)
        # aggr_attn = torch.cat(attention_maps).mean(dim=0)

        image = Image.open(img_path).convert("RGB").resize((image_size, image_size))
        
        if show_orig_img:
            attn_avg_vis, _ = show_cross_attention_plus_orig_img(
                                    caption_list, aggr_attn, orig_image=image, display_image=False)
        else:
            attn_avg_vis, _ = show_cross_attention(
                                    caption_list, aggr_attn, display_image=False)
            
        _, stack_attn=show_cross_attention_blackwhite(caption_list, aggr_attn, display_image=False)

        single_attn_paths=[]
        for (word_id, object_name) in keyword_ind:
            single_save_path=os.path.join(save_path, f'{object_name}_attn.png')
            Image.fromarray(stack_attn[word_id]).save(single_save_path)
            single_attn_paths.append(single_save_path)

        attn_save_path=os.path.join(save_path, f'attn_avg.png')
        attn_avg_vis.save(attn_save_path)
        return single_attn_paths


def mean_iou(seg_image_path, attn_image_path, Threshold=0.3):
    seg_image = Image.open(seg_image_path)
    seg_img_data = np.asarray(seg_image).astype(bool)
    seg_img_size=seg_image.size

    if len(seg_img_data.shape) == 3:
        seg_img_data=seg_img_data[:,:,-1]

    image = Image.open(attn_image_path)
    image=image.resize(seg_img_size)
    img_data = np.asarray(image).astype(np.float16)

    if len(img_data.shape) == 3:
        img_data=img_data[:,:,-1]

    img_data /=img_data.max()
    img_mask = img_data > Threshold

    intersection = np.logical_and(seg_img_data, img_mask).sum()
    union = np.logical_or(seg_img_data, img_mask).sum()

    return float(intersection/union)