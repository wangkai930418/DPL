import os
import json
import requests

from tqdm import tqdm
import argparse

from IPython.display import display
from clip_retrieval.clip_client import ClipClient, Modality
from PIL import Image

def download(embedding_url, local_filepath):
    r = requests.get(embedding_url)
    with open(local_filepath, "wb") as f:
        f.write(r.content)

def log_result(result, ind=0, folder_path=None):
    id, caption, url, similarity = result["id"], result["caption"], result["url"], result["similarity"]
    print(f"id: {id}")
    print(f"caption: {caption}")
    print(f"url: {url}")
    print(f"similarity: {similarity}")

    img = Image.open(requests.get(url, stream=True).raw)#.convert("RGB").resize((512, 512))
    # H,W = img.size
    width, height = img.size   # Get dimensions
    if min(width, height)<500:
        return
    
    new_width,new_height = min(width, height), min(width, height)

    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2
    
    img = img.crop((left, top, right, bottom))

    file_name=os.path.join(folder_path, f'{ind}.png')
    img.save(file_name)



def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_folder', type=str, default='clip_retrieval')
    parser.add_argument('--prompt_str', type=str, default='a horse and a goat')
    parser.add_argument('--num_images', type=int, default=500)

    args = parser.parse_args()
    return args



if __name__=="__main__":
    args = arguments()
    print(args)

    client = ClipClient(
        url="https://knn.laion.ai/knn-service",
        indice_name="laion5B-L-14",
        aesthetic_score=9,
        aesthetic_weight=0.5,
        modality=Modality.IMAGE,
        num_images=args.num_images,
    )

    search_text=args.prompt_str
    folder_name='_'.join(search_text.split(' '))
    folder_path = os.path.join(args.results_folder, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    cat_results = client.query(text=search_text)

    for ind in range(args.num_images):
        try:
            log_result(cat_results[ind], ind=ind, folder_path=folder_path)
        except:
            pass