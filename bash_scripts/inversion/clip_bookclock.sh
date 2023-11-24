
IMG_FORMAT='png'

IMG_FOLDER='clip_retrieval/'
SEARCH_TEXT='a_book_and_a_clock'

INDICES='2 5'

PLACEHOLDER1='<book-toy>'
PLACEHOLDER2='<clock-toy>'
PLACEHOLDER3='<books-toy>'
PLACEHOLDER5='<table-toy>'
# PLACEHOLDER4='<lamp-toy>'
# PLACEHOLDER6='<flower-toy>'

INIT_TOKEN1='book'
INIT_TOKEN2='clock'
INIT_TOKEN3='books'
INIT_TOKEN5='table'
# INIT_TOKEN6='flower'

MAX_ITER=0
LOSS='max'
ATTN_RES=16
ENTROPY_TEMPERATURE=1.0
COSINE_TEMPERATURE=1.0

### NOTE: ours
ATTN=1.0
ENTROPY=0.0
COSINE=0.0
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


for FILE_NAME in  1 21 48 64 76 97 130 168
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
    --placeholder_token ${PLACEHOLDER1} ${PLACEHOLDER2} ${PLACEHOLDER3}  ${PLACEHOLDER5}  \
    --initializer_token ${INIT_TOKEN1} ${INIT_TOKEN2} ${INIT_TOKEN3}  ${INIT_TOKEN5}  \
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