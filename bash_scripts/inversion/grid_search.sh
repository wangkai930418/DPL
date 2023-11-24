
FILE_NAME="catdog"

MAX_ITER=0
# PROMPT="a <cat-toy> and a <dog-toy>"

ATTN=1.0
ENTROPY=0.1
COSINE=0.2

### alpha
al_ATTN=27.6 
al_ENTROPY=25.0
al_COSINE=25.0

### beta
be_ATTN=0.3
be_ENTROPY=2.5
be_COSINE=0.9

for ENTROPY in 0.02 
do
for COSINE in 0.2
do
for al_ENTROPY in 25 75
do
for al_COSINE in 25 50 75
do
for be_ENTROPY in 2.5 5.0
do
for be_COSINE in 0.6 0.9
do

CUDA_VISIBLE_DEVICES=7 python _null_attend_textinv.py \
    --input_image "images/${FILE_NAME}.jpg" \
    --prompt_str "a <cat-toy> and a <dog-toy>" \
    --results_folder "output/${FILE_NAME}/" \
    --negative_guidance_scale 7.5 \
    --null_inner_steps 51 \
    --attn_inner_steps 21 \
    --indices_to_alter 2 5 \
    --lam_maxattn ${ATTN} \
    --lam_entropy ${ENTROPY} \
    --lam_cosine ${COSINE} \
    --print_freq 10 \
    --max_iter_to_alter ${MAX_ITER} \
    --placeholder_token '<cat-toy>' '<dog-toy>' \
    --initializer_token 'cat' 'dog' \
    --alpha_max  ${al_ATTN} \
    --alpha_ent  ${al_ENTROPY} \
    --alpha_cos  ${al_COSINE} \
    --beta_max  ${be_ATTN} \
    --beta_ent  ${be_ENTROPY} \
    --beta_cos  ${be_COSINE} \

done
done

done
done

done
done
