
al_COSINE=0.6
SEARCH_TEXT='a_book_and_a_clock'
IMG_FOLDER='clip_retrieval/'
IMG_FORMAT='png'

PLACEHOLDER1='<book-toy>'
PLACEHOLDER2='<clock-toy>'
# PLACEHOLDER3='<books-toy>'
# PLACEHOLDER4='<lamp-toy>'
# PLACEHOLDER5='<table-toy>'

INIT_TOKEN1='book'
INIT_TOKEN2='clock'
# INIT_TOKEN3='books'
# INIT_TOKEN4='lamp'
# INIT_TOKEN5='table'

NULL_STEP=51
ATTN_STEP=21

# FILE_LIST=$(ls -1 clip_retrieval/${SEARCH_TEXT} | sed -e 's/\.[^.]*$//')
# 1 21 48 64 76 97 130 168

for FILE_NAME in 1 21 48
do

echo "current image is ${FILE_NAME}.png"

CUDA_VISIBLE_DEVICES=1 \
python seg_null_textinv.py \
        --alpha_cos  ${al_COSINE} \
        --input_image ${IMG_FOLDER}/${SEARCH_TEXT}/${FILE_NAME}.${IMG_FORMAT} \
        --results_folder output/${SEARCH_TEXT}_${FILE_NAME}/ \
        --seg_dirs gsam_mask/${SEARCH_TEXT}_${FILE_NAME}/ \
        --negative_guidance_scale 7.5 \
        --null_inner_steps ${NULL_STEP} \
        --attn_inner_steps ${ATTN_STEP} \
        --print_freq 10 \
        --placeholder_token ${PLACEHOLDER1} ${PLACEHOLDER2}  \
        --initializer_token ${INIT_TOKEN1} ${INIT_TOKEN2}  \
        --smooth_op \
        --softmax_op \

done