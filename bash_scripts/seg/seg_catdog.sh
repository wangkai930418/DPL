
al_COSINE=0.6
SEARCH_TEXT='a_cat_and_a_dog'
IMG_FOLDER='clip_retrieval/'
IMG_FORMAT='png'

PLACEHOLDER1='<cat-toy>'
PLACEHOLDER2='<dog-toy>'

INIT_TOKEN1='cat'
INIT_TOKEN2='dog'

NULL_STEP=51
ATTN_STEP=21

FILE_LIST=$(ls -1 clip_retrieval/${SEARCH_TEXT} | sed -e 's/\.[^.]*$//')

for FILE_NAME in $FILE_LIST
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
        --placeholder_token ${PLACEHOLDER1} ${PLACEHOLDER2}   \
        --initializer_token ${INIT_TOKEN1} ${INIT_TOKEN2}   \
        --smooth_op \
        --softmax_op \

done