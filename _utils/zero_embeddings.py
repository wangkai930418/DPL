import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

def generate_captions(input_prompt):
    input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids.to("cuda")

    outputs = model.generate(
        input_ids, temperature=0.8, num_return_sequences=16, do_sample=True, max_new_tokens=128, top_k=10
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl", device_map="auto", torch_dtype=torch.float16)

source_concept = "wolf"
target_concept = "lion"

source_text = f"Provide a caption for images containing a {source_concept}. "
"The captions should be in English and should be no longer than 150 characters."

target_text = f"Provide a caption for images containing a {target_concept}. "
"The captions should be in English and should be no longer than 150 characters."

source_captions = generate_captions(source_text)
target_captions = generate_captions(target_concept)

from diffusers import StableDiffusionPix2PixZeroPipeline 

pipeline = StableDiffusionPix2PixZeroPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16
)
pipeline = pipeline.to("cuda")
tokenizer = pipeline.tokenizer
text_encoder = pipeline.text_encoder


import torch 

def embed_captions(sentences, tokenizer, text_encoder, device="cuda"):
    with torch.no_grad():
        embeddings = []
        for sent in sentences:
            text_inputs = tokenizer(
                sent,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(text_input_ids.to(device), attention_mask=None)[0]
            embeddings.append(prompt_embeds)
    return torch.concatenate(embeddings, dim=0).mean(dim=0).unsqueeze(0)

source_embeddings = embed_captions(source_captions, tokenizer, text_encoder)
target_embeddings = embed_captions(target_captions, tokenizer, text_encoder)
torch.save(source_embeddings, f'blip_embed/{source_concept}.pt')
torch.save(target_embeddings, f'blip_embed/{target_concept}.pt')
print('done')
