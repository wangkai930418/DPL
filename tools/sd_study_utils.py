from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm.notebook import tqdm
import torch

import math
from typing import List, Tuple, Union
from PIL import Image
import cv2
import numpy as np
import os
import re
import torch
from IPython.display import display
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from _utils.ptp_utils import *

import torchvision.transforms as transforms
from sklearn.decomposition import PCA
import pickle as pkl
import torch.nn.functional as F
import argparse
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score, v_measure_score

transform_train = transforms.Compose([
    transforms.ToPILImage(),
])
pca = PCA(n_components=3)


def save_mask(mask, output_name):
    mask_image = transform_train(mask.float())
    mask_image.save(output_name)

# def save_mask(mask, label,output_dir):
#     mask_image = transform_train(mask.float())
#     mask_image.save(os.path.join(output_dir,f"{label}.png"))

def show_image(image):
    image = 255 * image / image.max()
    image = image.unsqueeze(-1).expand(*image.shape, 3)
    image = image.numpy().astype(np.uint8)
    image = np.array(Image.fromarray(image).resize((256, 256)))
    return image

def cluster2noun_mod(clusters, background_segment_threshold, num_segments, nouns, cross_attention):
    REPEAT=clusters.shape[0]/cross_attention.shape[0]
    
    result = {}
    result_mask={}
    nouns_indices = [index for (index, word) in nouns]
    nouns_maps = cross_attention.cpu().numpy()[:, :, [i + 1 for i in nouns_indices]]
    nouns_maps = cross_attention.unsqueeze(-1).cpu().numpy()
    normalized_nouns_maps = np.zeros_like(nouns_maps).repeat(REPEAT, axis=0).repeat(REPEAT, axis=1)
    for i in range(nouns_maps.shape[-1]):
        curr_noun_map = nouns_maps[:, :, i].repeat(REPEAT, axis=0).repeat(REPEAT, axis=1)
        normalized_nouns_maps[:, :, i] = (curr_noun_map - np.abs(curr_noun_map.min())) / curr_noun_map.max()
    for c in range(num_segments):
        cluster_mask = np.zeros_like(clusters)
        cluster_mask[clusters == c] = 1
        score_maps = [cluster_mask * normalized_nouns_maps[:, :, i] for i in range(len(nouns_indices))]
        scores = [score_map.sum() / cluster_mask.sum() for score_map in score_maps]
        result[c] = nouns[np.argmax(np.array(scores))] if max(scores) > background_segment_threshold else "BG"
        result_mask[c]=cluster_mask
    return result, result_mask

def cluster2noun_(clusters, background_segment_threshold, num_segments, nouns, cross_attention):
    REPEAT=clusters.shape[0]/cross_attention.shape[0]
    
    result = {}
    result_mask={}
    nouns_indices = [index for (index, word) in nouns]
    nouns_maps = cross_attention.cpu().numpy()[:, :, [i + 1 for i in nouns_indices]]
    # nouns_maps = cross_attention.unsqueeze(-1).cpu().numpy()
    normalized_nouns_maps = np.zeros_like(nouns_maps).repeat(REPEAT, axis=0).repeat(REPEAT, axis=1)
    for i in range(nouns_maps.shape[-1]):
        curr_noun_map = nouns_maps[:, :, i].repeat(REPEAT, axis=0).repeat(REPEAT, axis=1)
        normalized_nouns_maps[:, :, i] = (curr_noun_map - np.abs(curr_noun_map.min())) / curr_noun_map.max()
    for c in range(num_segments):
        cluster_mask = np.zeros_like(clusters)
        cluster_mask[clusters == c] = 1
        score_maps = [cluster_mask * normalized_nouns_maps[:, :, i] for i in range(len(nouns_indices))]
        scores = [score_map.sum() / cluster_mask.sum() for score_map in score_maps]
        result[c] = nouns[np.argmax(np.array(scores))] if max(scores) > background_segment_threshold else "BG"
        result_mask[c]=cluster_mask
    return result, result_mask


def aggregate_attention( attention_maps, 
                        res: int, from_where: List[str], 
                        is_cross: bool, select: int, prompts,):
    out = []
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu(), attention_maps

def cluster(self_attention, num_segments=5,):
    np.random.seed(1)
    resolution, feat_dim = self_attention.shape[0], self_attention.shape[-1]
    attn = self_attention.cpu().numpy().reshape(resolution ** 2, feat_dim)
    kmeans = KMeans(n_clusters=num_segments, n_init=10).fit(attn)
    clusters = kmeans.labels_
    clusters = clusters.reshape(resolution, resolution)
    return clusters

def run_clusters(avg_dict, resolution, dict_key, save_path, special_name):
    clusters = cluster(avg_dict[dict_key][resolution], num_segments=5,)
    output_name=f'cluster_{dict_key}_{resolution}_{special_name}.png'
    npy_name=f'cluster_{dict_key}_{resolution}_{special_name}.npy'
    
    np.save(os.path.join(save_path, npy_name), clusters)

    plt.imshow(clusters)
    plt.axis('off')
    plt.savefig(os.path.join(save_path, output_name), bbox_inches='tight', pad_inches=0)

def read_pkl(path,):
    with open(path,'rb') as f:
        dict_ = pkl.load(f)
    return dict_


def draw_pca(avg_dict, resolution, dict_key, save_path, special_name):

    RESOLUTION=resolution
    # dict_key="key"

    if avg_dict[dict_key][RESOLUTION].__len__() == 0:
        return 
    before_pca = avg_dict[dict_key][RESOLUTION].reshape(RESOLUTION*RESOLUTION,-1).cpu().numpy()
    # print(before_pca.shape)

    pca.fit(before_pca)
    after_pca = pca.transform(before_pca)

    after_pca = after_pca.reshape(RESOLUTION,RESOLUTION,-1)
    pca_img_min = after_pca.min(axis=(0, 1))
    pca_img_max = after_pca.max(axis=(0, 1))
    pca_img = (after_pca - pca_img_min) / (pca_img_max - pca_img_min)

    output_name=f'pca_{dict_key}_{resolution}_{special_name}.png'
    pca_img = Image.fromarray((pca_img * 255).astype(np.uint8))
    pca_img=pca_img.resize((512,512))
    # display(show_image(torch.from_numpy(pca_img)))
    pca_img.save(os.path.join(save_path, output_name))
    # plt.imshow(pca_img)
    # plt.axis('off')
    # plt.savefig(os.path.join(save_path, output_name))

def image_normalize(numpy_array, save_path,output_name):
    numpy_array=numpy_array.cpu().numpy()
    img_min = numpy_array.min()
    img_max = numpy_array.max()
    normalize_array = (numpy_array - img_min) / (img_max - img_min)

    plt.imshow(normalize_array)
    plt.axis('off')
    plt.savefig(os.path.join(save_path, output_name), bbox_inches='tight', pad_inches=0)

    # normalize_array = Image.fromarray((normalize_array * 255).astype(np.uint8))
    # normalize_array=normalize_array.resize((512,512))
    # # normalize_array=view_images(normalize_array)
    # normalize_array.save(os.path.join(save_path, output_name))

def cross_cosine_with_name(resolution, inv_avg_dict, denoise_avg_dict, indice, save_path, save_crossattn=False, noun_name = ''):
    inv_cross_attn = inv_avg_dict['attn'][resolution][:,:,indice]
    denoise_cross_attn = denoise_avg_dict['attn'][resolution][:,:,indice]
    if save_crossattn:
        image_normalize(inv_cross_attn, save_path, f'crossattn_{resolution}_inv_{noun_name}.png')
        image_normalize(denoise_cross_attn, save_path, f'crossattn_{resolution}_denoise_{noun_name}.png')

    return F.cosine_similarity(inv_cross_attn.reshape(1,-1), denoise_cross_attn.reshape(1,-1))

def cross_cosine(resolution, inv_avg_dict, denoise_avg_dict, indice, save_path, save_crossattn=False,):
    inv_cross_attn = inv_avg_dict['attn'][resolution][:,:,indice]
    denoise_cross_attn = denoise_avg_dict['attn'][resolution][:,:,indice]
    if save_crossattn:
        image_normalize(inv_cross_attn, save_path, f'crossattn_{resolution}_inv.png')
        image_normalize(denoise_cross_attn, save_path, f'crossattn_{resolution}_denoise.png')

    return F.cosine_similarity(inv_cross_attn.reshape(1,-1), denoise_cross_attn.reshape(1,-1))

def save_crossattn(input_path, caption, inv_cross_avg_dict, denoise_cross_avg_dict, results_folder, RES=16):
    org_image = Image.open(input_path).convert("RGB")
    prompts=["<|startoftext|>",] + caption.split(' ') + ["<|endoftext|>",]
    inv_crossattn = inv_cross_avg_dict['attn'][RES]
    denoise_crossattn = denoise_cross_avg_dict['attn'][RES]

    # inv_crossattn_maps = [item.unsqueeze(0) for item in inv_crossattn]
    # denoise_crossattn_maps = [item.unsqueeze(0) for item in denoise_crossattn]

    # inv_crossattn_maps = torch.cat(inv_crossattn_maps).mean(dim=0)
    # denoise_crossattn_maps = torch.cat(denoise_crossattn_maps).mean(dim=0)

    attn_img1, _ = show_cross_attention_plus_orig_img(prompts, inv_crossattn, orig_image=org_image)
    attn_img1_wo, _ =show_cross_attention(prompts, inv_crossattn, )
    attn_img1_blakcwhite, _ =show_cross_attention_blackwhite(prompts, inv_crossattn)

    attn_img2, _ = show_cross_attention_plus_orig_img(prompts, denoise_crossattn, orig_image=org_image)
    attn_img2_wo, _ =show_cross_attention(prompts, denoise_crossattn, )
    attn_img2_blakcwhite, _ =show_cross_attention_blackwhite(prompts, denoise_crossattn)

    attn_img1.save(os.path.join(results_folder,'crossattn_inv.png')) 
    attn_img1_wo.save(os.path.join(results_folder,'crossattn_inv_wo.png')) 
    attn_img1_blakcwhite.save(os.path.join(results_folder,'crossattn_inv_blackwhite.png')) 

    attn_img2.save(os.path.join(results_folder,'crossattn_denoise.png')) 
    attn_img2_wo.save(os.path.join(results_folder,'crossattn_denoise_wo.png')) 
    attn_img2_blakcwhite.save(os.path.join(results_folder,'crossattn_denoise_blackwhite.png')) 