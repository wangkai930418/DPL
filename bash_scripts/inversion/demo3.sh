
# FILE_NAME="catdog"
FILE_NAME="lexica_kitten_puppy"
# IMG_FORMAT='png'
IMG_FORMAT='jpg'

PROMPT="a <kitten-toy> and a <puppy-toy>"
INDICES='2 5'
PLACEHOLDER1='<kitten-toy>'
PLACEHOLDER2='<puppy-toy>'
INIT_TOKEN1='kitten'
INIT_TOKEN2='puppy'

MAX_ITER=0
LOSS='max'
ATTN_RES=16
ENTROPY_TEMPERATURE=1.0
COSINE_TEMPERATURE=1.0

ATTN=1.0
ENTROPY=0.0
COSINE=0.2
DIFFERENCE=0.0

### alpha
al_ATTN=27.6 
al_ENTROPY=27.6 
al_COSINE=25.0

### beta
be_ATTN=0.3
be_ENTROPY=2.5
be_COSINE=0.9


CUDA_VISIBLE_DEVICES=2 python _null_attend_textinv.py \
    --input_image "images/${FILE_NAME}.${IMG_FORMAT}" \
    --prompt_str "${PROMPT}" \
    --results_folder "output/${FILE_NAME}/" \
    --negative_guidance_scale 7.5 \
    --null_inner_steps 51 \
    --attn_inner_steps 21 \
    --indices_to_alter ${INDICES} \
    --lam_maxattn ${ATTN} \
    --lam_entropy ${ENTROPY} \
    --lam_cosine ${COSINE} \
    --lam_difference ${DIFFERENCE} \
    --print_freq 10 \
    --max_iter_to_alter ${MAX_ITER} \
    --placeholder_token ${PLACEHOLDER1} ${PLACEHOLDER2} \
    --initializer_token ${INIT_TOKEN1} ${INIT_TOKEN2} \
    --alpha_max  ${al_ATTN} \
    --alpha_ent  ${al_ENTROPY} \
    --alpha_cos  ${al_COSINE} \
    --beta_max  ${be_ATTN} \
    --beta_ent  ${be_ENTROPY} \
    --beta_cos  ${be_COSINE} \
    --loss_type ${LOSS} \
    --attn_res ${ATTN_RES} \
    --entropy_T ${ENTROPY_TEMPERATURE} \
    --cosine_T ${COSINE_TEMPERATURE} \
    --smooth_op \
    --softmax_op \