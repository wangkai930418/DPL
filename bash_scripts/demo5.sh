
# SEARCH_TEXT='a_cat_and_a_dog'
IMG_FOLDER='images/'
IMG_FORMAT='jpg'
FILE_NAME='catdog'

PLACEHOLDER1='<cat-toy>'
PLACEHOLDER2='<dog-toy>'

INIT_TOKEN1='cat'
INIT_TOKEN2='dog'

NULL_STEP=31
ATTN_STEP=11

al_COSINE=25.0
be_COSINE=0.4

# FILE_LIST=$(ls -1 clip_retrieval/${SEARCH_TEXT} | sed -e 's/\.[^.]*$//')

for be_COSINE in 0.1 0.05 0.1 

do

echo "current image is ${FILE_NAME}.png"

CUDA_VISIBLE_DEVICES=5 \
python _seg_null_textinv.py \
        --input_image ${IMG_FOLDER}/${FILE_NAME}.${IMG_FORMAT} \
        --results_folder output/${FILE_NAME}/ \
        --seg_dirs seg_dirs/${FILE_NAME}/ \
        --negative_guidance_scale 7.5 \
        --print_freq 10 \
        --placeholder_token ${PLACEHOLDER1} ${PLACEHOLDER2}   \
        --initializer_token ${INIT_TOKEN1} ${INIT_TOKEN2}   \
        --smooth_op \
        --softmax_op \
        --alpha_cos  ${al_COSINE} \
        --beta_cos ${be_COSINE} \
        --null_inner_steps ${NULL_STEP} \
        --attn_inner_steps ${ATTN_STEP} \

done