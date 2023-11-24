
IMG_FORMAT='png'

IMG_FOLDER='clip_retrieval/'
SEARCH_TEXT='a_clock_and_a_book'

INDICES='2 5'

PLACEHOLDER1='<book-toy>'
PLACEHOLDER2='<clock-toy>'
PLACEHOLDER3='<books-toy>'
PLACEHOLDER4='<table-toy>'
PLACEHOLDER5='<chair-toy>'
PLACEHOLDER6='<watch-toy>'

INIT_TOKEN1='book'
INIT_TOKEN2='clock'
INIT_TOKEN3='books'
INIT_TOKEN4='table'
INIT_TOKEN5='chair'
INIT_TOKEN6='watch'

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


for FILE_NAME in 1 10 107 111 114 123 124 127 132 17 179 182 183 188 193 196 201 219 228 23 27 33 46 49 51 59 64 66 80 81 89 93 95 96
do

CUDA_VISIBLE_DEVICES=7 python _null_attend_textinv.py \
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
    --placeholder_token ${PLACEHOLDER1} ${PLACEHOLDER2} ${PLACEHOLDER3} ${PLACEHOLDER4} ${PLACEHOLDER5} ${PLACEHOLDER6} \
    --initializer_token ${INIT_TOKEN1} ${INIT_TOKEN2} ${INIT_TOKEN3} ${INIT_TOKEN4} ${INIT_TOKEN5} ${INIT_TOKEN6} \
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