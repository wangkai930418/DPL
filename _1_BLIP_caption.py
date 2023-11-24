from PIL import Image
import requests
from transformers import AutoProcessor, BlipForConditionalGeneration
import torch
import argparse
import os
import warnings
warnings.filterwarnings("ignore")

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=str, default=None)
    parser.add_argument('--results_folder', type=str, default='output/')
    parser.add_argument('--prompt_str', type=str, default='')
    parser.add_argument('--write_file', action='store_true')
    parser.add_argument('--no-write_file', dest='write_file', action='store_false')
    parser.set_defaults(write_file=True)

    args = parser.parse_args()
    return args



if __name__=="__main__":
    args = arguments()

    if os.path.isfile(args.input_image):
        dirs=[args.input_image]
        dirname=os.path.dirname(args.input_image)

    elif os.path.isdir(args.input_image):
        dirs = (os.listdir(args.input_image))
        dirname=args.input_image
        dirs = [os.path.join(dirname, dir) for dir in dirs]
        dirs.sort()

    print(f'The image base is {dirname}')
    print('\n'.join(dirs))

    text = args.prompt_str
    img_ids=[]
    WRITE2FILE=args.write_file

    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    for img_path in dirs[:]:
        print(img_path)
        
        if os.path.isdir(args.input_image):
            search_text = img_path.split('/')[-2]
            img_id = img_path.split('/')[-1].split('.')[0]
            _results_folder = os.path.join(args.results_folder, f"{search_text}_{img_id}")
        elif os.path.isfile(args.input_image):
            img_id = img_path.split('/')[-1].split('.')[0]
            _results_folder = os.path.join(args.results_folder, f"{img_id}")

        img_ids.append(img_id)
        print(_results_folder)
        if WRITE2FILE:
            os.makedirs(_results_folder, exist_ok=True)
        
        image = Image.open(img_path)
        inputs = processor(images=image, text=text, return_tensors="pt")

        generated_ids = model.generate(**inputs)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        print(generated_text)
        
        if WRITE2FILE:
            with open(os.path.join(_results_folder, f"prompt.txt"), "w") as f:
                f.write(generated_text)

    print(' '.join(img_ids))
