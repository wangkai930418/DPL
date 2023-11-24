
SEARCH_TEXT='a_cat_and_a_dog'
IMG_FOLDER='clip_retrieval/'
IMG_FORMAT='png'

PLACEHOLDER1='<cat-toy>'
PLACEHOLDER2='<dog-toy>'

INIT_TOKEN1='cat'
INIT_TOKEN2='dog'

NULL_STEP=51
ATTN_STEP=21

al_COSINE=0.6
be_COSINE=0.1

# FILE_LIST=$(ls -1 clip_retrieval/${SEARCH_TEXT} | sed -e 's/\.[^.]*$//')
FILE_LIST="0 1 2 3 6"

for al_COSINE in 0.9 0.6
do
for be_COSINE in 0.1 0.2 0.05
do

for FILE_NAME in $FILE_LIST
do

echo "current image is ${FILE_NAME}.png"

CUDA_VISIBLE_DEVICES=7 \
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
done
done