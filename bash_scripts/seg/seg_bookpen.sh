
# al_COSINE=0.6
al_COSINE=0.7
SEARCH_TEXT='a_book_and_a_pen'
IMG_FOLDER='clip_retrieval/'
IMG_FORMAT='png'

PLACEHOLDER1='<book-toy>'
PLACEHOLDER2='<pen-toy>'
PLACEHOLDER3='<notebook-toy>'
# PLACEHOLDER4='<woman-toy>'

INIT_TOKEN1='book'
INIT_TOKEN2='pen'
INIT_TOKEN3='notebook'
# INIT_TOKEN4='woman'

NULL_STEP=51
ATTN_STEP=21

FILE_LIST=$(ls -1 clip_retrieval/${SEARCH_TEXT} | sed -e 's/\.[^.]*$//')

for FILE_NAME in $FILE_LIST
# for FILE_NAME in 103 107 110 118 137 148 159 20 23 25 30 34 35 45 60 67 7 73 80 9
do

echo "current image is ${FILE_NAME}.png"

CUDA_VISIBLE_DEVICES=6 \
python seg_null_textinv.py \
        --alpha_cos  ${al_COSINE} \
        --input_image ${IMG_FOLDER}/${SEARCH_TEXT}/${FILE_NAME}.${IMG_FORMAT} \
        --results_folder output/${SEARCH_TEXT}_${FILE_NAME}/ \
        --seg_dirs gsam_mask/${SEARCH_TEXT}_${FILE_NAME}/ \
        --negative_guidance_scale 7.5 \
        --null_inner_steps ${NULL_STEP} \
        --attn_inner_steps ${ATTN_STEP} \
        --print_freq 10 \
        --placeholder_token ${PLACEHOLDER1} ${PLACEHOLDER2}  ${PLACEHOLDER3} \
        --initializer_token ${INIT_TOKEN1} ${INIT_TOKEN2}  ${INIT_TOKEN3}  \
        --smooth_op \
        --softmax_op \

done