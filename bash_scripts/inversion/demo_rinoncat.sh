
IMG_FORMAT='jpg'
FILE_NAME='rinon_cat'
IMG_FOLDER='images/'
# SEARCH_TEXT='a_cat_and_a_dog'

### NOTE: ablate 1
PLACEHOLDER1='<cat-toy>'
PLACEHOLDER2='<basket-toy>'

INIT_TOKEN1='cat'
INIT_TOKEN2='basket'

MAX_ITER=0
LOSS='max'
ATTN_RES=16
ENTROPY_TEMPERATURE=1.0
COSINE_TEMPERATURE=1.0

### NOTE: ours
### NOTE: ablate 2
ATTN=1.0
ENTROPY=0.0
COSINE=0.2
DIFFERENCE=0.0
NULL_STEP=51
ATTN_STEP=21

### NOTE: null-text inversion
# ATTN=0.0
# ENTROPY=0.0
# COSINE=0.0
# DIFFERENCE=0.0
# NULL_STEP=51
# ATTN_STEP=1

### alpha
### NOTE: ablate 3
al_ATTN=27.6 
al_ENTROPY=27.6 
al_COSINE=25.0

### beta
### NOTE: ablate 4
be_ATTN=0.3
be_ENTROPY=2.5
be_COSINE=0.9

# for FILE_NAME in 0 3 6 9 21 42 50 51 54 55 58 61 63 64
# do

CUDA_VISIBLE_DEVICES=1 python _null_attend_textinv.py \
    --input_image ${IMG_FOLDER}/${FILE_NAME}.${IMG_FORMAT} \
    --results_folder output/${FILE_NAME}/ \
    --negative_guidance_scale 7.5 \
    --null_inner_steps ${NULL_STEP} \
    --attn_inner_steps ${ATTN_STEP} \
    --lam_maxattn ${ATTN} \
    --lam_entropy ${ENTROPY} \
    --lam_cosine ${COSINE} \
    --lam_difference ${DIFFERENCE} \
    --print_freq 10 \
    --max_iter_to_alter ${MAX_ITER} \
    --placeholder_token ${PLACEHOLDER1} ${PLACEHOLDER2}   \
    --initializer_token ${INIT_TOKEN1} ${INIT_TOKEN2}   \
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
    # --indices_to_alter ${INDICES} \
    # --prompt_str "${PROMPT}" \
    
# done