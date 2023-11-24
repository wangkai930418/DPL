
IMG_FORMAT='png'

IMG_FOLDER='clip_retrieval/'
SEARCH_TEXT='a_sheep_and_a_horse'

INDICES='2 5'

PLACEHOLDER1='<sheep-toy>'
PLACEHOLDER2='<horse-toy>'
PLACEHOLDER3='<girl-toy>'
PLACEHOLDER4='<woman-toy>'
# PLACEHOLDER5='<chair-toy>'
# PLACEHOLDER6='<watch-toy>'

INIT_TOKEN1='sheep'
INIT_TOKEN2='horse'
INIT_TOKEN3='girl'
INIT_TOKEN4='woman'
# INIT_TOKEN5='chair'
# INIT_TOKEN6='watch'

MAX_ITER=0
LOSS='max'
ATTN_RES=16
ENTROPY_TEMPERATURE=1.0
COSINE_TEMPERATURE=1.0

### NOTE: ours
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
al_ATTN=27.6 
al_ENTROPY=27.6 
al_COSINE=25.0

### beta
be_ATTN=0.3
be_ENTROPY=2.5
be_COSINE=0.9


for FILE_NAME in 10 141 218 261 29 297 299 3 61 71 89
do

CUDA_VISIBLE_DEVICES=4 python _null_attend_textinv.py \
    --input_image ${IMG_FOLDER}/${SEARCH_TEXT}/${FILE_NAME}.${IMG_FORMAT} \
    --results_folder output/${SEARCH_TEXT}_${FILE_NAME}/ \
    --negative_guidance_scale 7.5 \
    --null_inner_steps ${NULL_STEP} \
    --attn_inner_steps ${ATTN_STEP} \
    --lam_maxattn ${ATTN} \
    --lam_entropy ${ENTROPY} \
    --lam_cosine ${COSINE} \
    --lam_difference ${DIFFERENCE} \
    --print_freq 10 \
    --max_iter_to_alter ${MAX_ITER} \
    --placeholder_token ${PLACEHOLDER1} ${PLACEHOLDER2} ${PLACEHOLDER3} ${PLACEHOLDER4}  \
    --initializer_token ${INIT_TOKEN1} ${INIT_TOKEN2} ${INIT_TOKEN3} ${INIT_TOKEN4}   \
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
    # --prompt_str "${PROMPT}" \
    # --indices_to_alter ${INDICES} \

done