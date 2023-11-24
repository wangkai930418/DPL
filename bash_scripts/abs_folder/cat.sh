
IMG_FORMAT='jpg'

IMG_FOLDER='coco_seginv/'


OBJECT='cat'
SEARCH_TEXT='val2014/'

PLACEHOLDER1='<cat-toy>'

INIT_TOKEN1='cat'


NULL_STEP=31
ATTN_STEP=31

### NOTE: Ablation study parameters
al_COSINE=50.0
al_IOU=25.0

be_COSINE=0.7
be_IOU=0.7

lam_COSINE=1.0
lam_IOU=1.0

### NOTE: not ablate for numbers
al_KL=25.0
al_SIM=25.0
al_ADJ=50.0

be_KL=1.0
be_SIM=0.9
be_ADJ=0.1

# lam_KL=1.0
lam_KL=0.0
lam_SIM=0.0
lam_ADJ=0.0
# lam_ADJ=2.0

GUIDE=7.5

FILE_LIST=$(ls -1 ${IMG_FOLDER}/${OBJECT}/${SEARCH_TEXT} | sed -e 's/\.[^.]*$//')
echo ${FILE_LIST}


# for GUIDE in 3.0 5.0 7.5
# do

for FILE_NAME in $FILE_LIST
do

### NOTE: ablate 1
### NOTE: Ablation study parameters
al_COSINE=50.0
al_IOU=25.0

be_COSINE=0.7
be_IOU=0.7

lam_COSINE=1.0
lam_IOU=1.0


for al_COSINE in 50.0 25.0 75.0 10.0
do

echo "current image is ${FILE_NAME}.png"

CUDA_VISIBLE_DEVICES=6 \
python _3_seg_null_textinv.py \
        --input_image ${IMG_FOLDER}/${OBJECT}/${SEARCH_TEXT}/${FILE_NAME}/${FILE_NAME}.${IMG_FORMAT} \
        --results_folder ${IMG_FOLDER}/${OBJECT}/${SEARCH_TEXT}/${FILE_NAME}/ \
        --seg_dirs ${IMG_FOLDER}/${OBJECT}/${SEARCH_TEXT}/${FILE_NAME}/ \
        --negative_guidance_scale ${GUIDE} \
        --print_freq 10 \
        --placeholder_token ${PLACEHOLDER1}    \
        --initializer_token ${INIT_TOKEN1}   \
        --smooth_op \
        --softmax_op \
        --alpha_cos  ${al_COSINE} \
        --alpha_iou  ${al_IOU} \
        --alpha_kl  ${al_KL} \
        --alpha_sim  ${al_SIM} \
        --alpha_adj  ${al_ADJ} \
        --beta_cos ${be_COSINE} \
        --beta_iou ${be_IOU} \
        --beta_kl ${be_KL} \
        --beta_sim ${be_SIM} \
        --beta_adj ${be_ADJ} \
        --lam_cos ${lam_COSINE} \
        --lam_iou ${lam_IOU} \
        --lam_kl ${lam_KL} \
        --lam_sim ${lam_SIM} \
        --lam_adj ${lam_ADJ} \
        --null_inner_steps ${NULL_STEP} \
        --attn_inner_steps ${ATTN_STEP} \
        --no-adj_bind \
        # --adj_bind \

done


### NOTE: ablate 2
### NOTE: Ablation study parameters
al_COSINE=50.0
al_IOU=25.0

be_COSINE=0.7
be_IOU=0.7

lam_COSINE=1.0
lam_IOU=1.0


for al_IOU in 50.0 75.0 10.0
do

echo "current image is ${FILE_NAME}.png"

CUDA_VISIBLE_DEVICES=6 \
python _3_seg_null_textinv.py \
        --input_image ${IMG_FOLDER}/${OBJECT}/${SEARCH_TEXT}/${FILE_NAME}/${FILE_NAME}.${IMG_FORMAT} \
        --results_folder ${IMG_FOLDER}/${OBJECT}/${SEARCH_TEXT}/${FILE_NAME}/ \
        --seg_dirs ${IMG_FOLDER}/${OBJECT}/${SEARCH_TEXT}/${FILE_NAME}/ \
        --negative_guidance_scale ${GUIDE} \
        --print_freq 10 \
        --placeholder_token ${PLACEHOLDER1}    \
        --initializer_token ${INIT_TOKEN1}   \
        --smooth_op \
        --softmax_op \
        --alpha_cos  ${al_COSINE} \
        --alpha_iou  ${al_IOU} \
        --alpha_kl  ${al_KL} \
        --alpha_sim  ${al_SIM} \
        --alpha_adj  ${al_ADJ} \
        --beta_cos ${be_COSINE} \
        --beta_iou ${be_IOU} \
        --beta_kl ${be_KL} \
        --beta_sim ${be_SIM} \
        --beta_adj ${be_ADJ} \
        --lam_cos ${lam_COSINE} \
        --lam_iou ${lam_IOU} \
        --lam_kl ${lam_KL} \
        --lam_sim ${lam_SIM} \
        --lam_adj ${lam_ADJ} \
        --null_inner_steps ${NULL_STEP} \
        --attn_inner_steps ${ATTN_STEP} \
        --no-adj_bind \
        # --adj_bind \

done

### NOTE: ablate 3
### NOTE: Ablation study parameters
al_COSINE=50.0
al_IOU=25.0

be_COSINE=0.7
be_IOU=0.7

lam_COSINE=1.0
lam_IOU=1.0


for be_COSINE in 0.3 0.5 0.9
do

echo "current image is ${FILE_NAME}.png"

CUDA_VISIBLE_DEVICES=6 \
python _3_seg_null_textinv.py \
        --input_image ${IMG_FOLDER}/${OBJECT}/${SEARCH_TEXT}/${FILE_NAME}/${FILE_NAME}.${IMG_FORMAT} \
        --results_folder ${IMG_FOLDER}/${OBJECT}/${SEARCH_TEXT}/${FILE_NAME}/ \
        --seg_dirs ${IMG_FOLDER}/${OBJECT}/${SEARCH_TEXT}/${FILE_NAME}/ \
        --negative_guidance_scale ${GUIDE} \
        --print_freq 10 \
        --placeholder_token ${PLACEHOLDER1}    \
        --initializer_token ${INIT_TOKEN1}   \
        --smooth_op \
        --softmax_op \
        --alpha_cos  ${al_COSINE} \
        --alpha_iou  ${al_IOU} \
        --alpha_kl  ${al_KL} \
        --alpha_sim  ${al_SIM} \
        --alpha_adj  ${al_ADJ} \
        --beta_cos ${be_COSINE} \
        --beta_iou ${be_IOU} \
        --beta_kl ${be_KL} \
        --beta_sim ${be_SIM} \
        --beta_adj ${be_ADJ} \
        --lam_cos ${lam_COSINE} \
        --lam_iou ${lam_IOU} \
        --lam_kl ${lam_KL} \
        --lam_sim ${lam_SIM} \
        --lam_adj ${lam_ADJ} \
        --null_inner_steps ${NULL_STEP} \
        --attn_inner_steps ${ATTN_STEP} \
        --no-adj_bind \
        # --adj_bind \

done


### NOTE: ablate 4
### NOTE: Ablation study parameters
al_COSINE=50.0
al_IOU=25.0

be_COSINE=0.7
be_IOU=0.7

lam_COSINE=1.0
lam_IOU=1.0


for be_IOU in 0.3 0.5 0.9
do

echo "current image is ${FILE_NAME}.png"

CUDA_VISIBLE_DEVICES=6 \
python _3_seg_null_textinv.py \
        --input_image ${IMG_FOLDER}/${OBJECT}/${SEARCH_TEXT}/${FILE_NAME}/${FILE_NAME}.${IMG_FORMAT} \
        --results_folder ${IMG_FOLDER}/${OBJECT}/${SEARCH_TEXT}/${FILE_NAME}/ \
        --seg_dirs ${IMG_FOLDER}/${OBJECT}/${SEARCH_TEXT}/${FILE_NAME}/ \
        --negative_guidance_scale ${GUIDE} \
        --print_freq 10 \
        --placeholder_token ${PLACEHOLDER1}    \
        --initializer_token ${INIT_TOKEN1}   \
        --smooth_op \
        --softmax_op \
        --alpha_cos  ${al_COSINE} \
        --alpha_iou  ${al_IOU} \
        --alpha_kl  ${al_KL} \
        --alpha_sim  ${al_SIM} \
        --alpha_adj  ${al_ADJ} \
        --beta_cos ${be_COSINE} \
        --beta_iou ${be_IOU} \
        --beta_kl ${be_KL} \
        --beta_sim ${be_SIM} \
        --beta_adj ${be_ADJ} \
        --lam_cos ${lam_COSINE} \
        --lam_iou ${lam_IOU} \
        --lam_kl ${lam_KL} \
        --lam_sim ${lam_SIM} \
        --lam_adj ${lam_ADJ} \
        --null_inner_steps ${NULL_STEP} \
        --attn_inner_steps ${ATTN_STEP} \
        --no-adj_bind \
        # --adj_bind \

done

### NOTE: ablate 5
### NOTE: Ablation study parameters
al_COSINE=50.0
al_IOU=25.0

be_COSINE=0.7
be_IOU=0.7

lam_COSINE=1.0
lam_IOU=1.0


for lam_COSINE in 0.1 0.5 2.0 5.0
do

echo "current image is ${FILE_NAME}.png"

CUDA_VISIBLE_DEVICES=6 \
python _3_seg_null_textinv.py \
        --input_image ${IMG_FOLDER}/${OBJECT}/${SEARCH_TEXT}/${FILE_NAME}/${FILE_NAME}.${IMG_FORMAT} \
        --results_folder ${IMG_FOLDER}/${OBJECT}/${SEARCH_TEXT}/${FILE_NAME}/ \
        --seg_dirs ${IMG_FOLDER}/${OBJECT}/${SEARCH_TEXT}/${FILE_NAME}/ \
        --negative_guidance_scale ${GUIDE} \
        --print_freq 10 \
        --placeholder_token ${PLACEHOLDER1}    \
        --initializer_token ${INIT_TOKEN1}   \
        --smooth_op \
        --softmax_op \
        --alpha_cos  ${al_COSINE} \
        --alpha_iou  ${al_IOU} \
        --alpha_kl  ${al_KL} \
        --alpha_sim  ${al_SIM} \
        --alpha_adj  ${al_ADJ} \
        --beta_cos ${be_COSINE} \
        --beta_iou ${be_IOU} \
        --beta_kl ${be_KL} \
        --beta_sim ${be_SIM} \
        --beta_adj ${be_ADJ} \
        --lam_cos ${lam_COSINE} \
        --lam_iou ${lam_IOU} \
        --lam_kl ${lam_KL} \
        --lam_sim ${lam_SIM} \
        --lam_adj ${lam_ADJ} \
        --null_inner_steps ${NULL_STEP} \
        --attn_inner_steps ${ATTN_STEP} \
        --no-adj_bind \
        # --adj_bind \

done


done
