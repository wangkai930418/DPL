import math
from typing import List, Tuple, Union
from PIL import Image
import cv2
import numpy as np
import torch
from IPython.display import display

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
    images = []
    split_imgs = []
    for i in range(len(prompts)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
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
    images = []
    split_imgs = []
    white_image = Image.new('RGB', (500, 500), (255, 255, 255))
    
    for i in range(len(prompts)):
        image = attention_maps[:, :, i]
        image = show_image_relevance(image, white_image)
        
        image = image.astype(np.uint8)
        
        image = np.array(Image.fromarray(image).resize((256, 256)))
        split_imgs.append(image)
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
    
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam

    image = image.resize((relevnace_res ** 2, relevnace_res ** 2))
    image = np.array(image)

    image_relevance = image_relevance.reshape(1, 1, image_relevance.shape[-1], image_relevance.shape[-1])
    image_relevance = image_relevance.cuda() 
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=relevnace_res ** 2, mode='bilinear')
    image_relevance = image_relevance.cpu() 
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