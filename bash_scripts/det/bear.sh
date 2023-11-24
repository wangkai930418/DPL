IMG_FORMAT='jpg'

IMG_FOLDER='coco_seginv/'


OBJECT='bear'
SEARCH_TEXT='val2014/'

PLACEHOLDER1='<bear-toy>'

INIT_TOKEN1='bear'

NULL_STEP=31
ATTN_STEP=31


### NOTE: not ablate for numbers
al_KL=25.0
al_SIM=25.0
al_ADJ=50.0

be_KL=1.0
be_SIM=0.9
be_ADJ=0.1

lam_KL=0.0
lam_SIM=0.0
lam_ADJ=0.0

GUIDE=7.5

### NOTE: Ablation study parameters
al_COSINE=25.0
al_IOU=25.0  
be_COSINE=0.5 
be_IOU=0.5
lam_COSINE=0.1
lam_IOU=1.0


FILE_LIST=$(ls -1 ${IMG_FOLDER}/${OBJECT}/${SEARCH_TEXT} | sed -e 's/\.[^.]*$//')
echo ${FILE_LIST}


for FILE_NAME in $FILE_LIST
do
### NOTE: ablate 1: 25.0 10.0 50.0 similar

for al_COSINE in 50.0 25.0 10.0 
do

echo "current image is ${FILE_NAME}.${IMG_FORMAT}"

CUDA_VISIBLE_DEVICES=5 \
python _3_det_null_textinv.py \
        --input_image ${IMG_FOLDER}/${OBJECT}/${SEARCH_TEXT}/${FILE_NAME}/${FILE_NAME}.${IMG_FORMAT} \
        --results_folder ${IMG_FOLDER}/${OBJECT}/${SEARCH_TEXT}/${FILE_NAME}/ \
        --seg_dirs ${IMG_FOLDER}/${OBJECT}/${SEARCH_TEXT}/${FILE_NAME}/ \
        --negative_guidance_scale 7.5 \
        --print_freq 10 \
        --placeholder_token ${PLACEHOLDER1}    \
        --initializer_token ${INIT_TOKEN1}   \
        --smooth_op \
        --softmax_op \
        --alpha_cos  ${al_COSINE} \
        --alpha_iou  ${al_IOU} \
        --alpha_kl  ${al_KL} \
        --alpha_sim  ${al_SIM} \
        --beta_cos ${be_COSINE} \
        --beta_iou ${be_IOU} \
        --beta_kl ${be_KL} \
        --beta_sim ${be_SIM} \
        --lam_cos ${lam_COSINE} \
        --lam_iou ${lam_IOU} \
        --lam_kl ${lam_KL} \
        --lam_sim ${lam_SIM} \
        --null_inner_steps ${NULL_STEP} \
        --attn_inner_steps ${ATTN_STEP} \

done


### NOTE: ablate 2
### NOTE: Ablation study parameters
al_COSINE=25.0
al_IOU=25.0  
be_COSINE=0.5 
be_IOU=0.5
lam_COSINE=0.1
lam_IOU=1.0

### NOTE: 25.0 > 10.0 50.0

for al_IOU in 10.0 50.0
do

echo "current image is ${FILE_NAME}.${IMG_FORMAT}"

CUDA_VISIBLE_DEVICES=5 \
python _3_det_null_textinv.py \
        --input_image ${IMG_FOLDER}/${OBJECT}/${SEARCH_TEXT}/${FILE_NAME}/${FILE_NAME}.${IMG_FORMAT} \
        --results_folder ${IMG_FOLDER}/${OBJECT}/${SEARCH_TEXT}/${FILE_NAME}/ \
        --seg_dirs ${IMG_FOLDER}/${OBJECT}/${SEARCH_TEXT}/${FILE_NAME}/ \
        --negative_guidance_scale 7.5 \
        --print_freq 10 \
        --placeholder_token ${PLACEHOLDER1}    \
        --initializer_token ${INIT_TOKEN1}   \
        --smooth_op \
        --softmax_op \
        --alpha_cos  ${al_COSINE} \
        --alpha_iou  ${al_IOU} \
        --alpha_kl  ${al_KL} \
        --alpha_sim  ${al_SIM} \
        --beta_cos ${be_COSINE} \
        --beta_iou ${be_IOU} \
        --beta_kl ${be_KL} \
        --beta_sim ${be_SIM} \
        --lam_cos ${lam_COSINE} \
        --lam_iou ${lam_IOU} \
        --lam_kl ${lam_KL} \
        --lam_sim ${lam_SIM} \
        --null_inner_steps ${NULL_STEP} \
        --attn_inner_steps ${ATTN_STEP} \

done



### NOTE: ablate 3
### NOTE: Ablation study parameters
al_COSINE=25.0
al_IOU=25.0  
be_COSINE=0.5 
be_IOU=0.5
lam_COSINE=0.1
lam_IOU=1.0

### NOTE: 0.3 0.5 0.7 similar

for be_COSINE in 0.3 0.7
do

echo "current image is ${FILE_NAME}.${IMG_FORMAT}"

CUDA_VISIBLE_DEVICES=5 \
python _3_det_null_textinv.py \
        --input_image ${IMG_FOLDER}/${OBJECT}/${SEARCH_TEXT}/${FILE_NAME}/${FILE_NAME}.${IMG_FORMAT} \
        --results_folder ${IMG_FOLDER}/${OBJECT}/${SEARCH_TEXT}/${FILE_NAME}/ \
        --seg_dirs ${IMG_FOLDER}/${OBJECT}/${SEARCH_TEXT}/${FILE_NAME}/ \
        --negative_guidance_scale 7.5 \
        --print_freq 10 \
        --placeholder_token ${PLACEHOLDER1}    \
        --initializer_token ${INIT_TOKEN1}   \
        --smooth_op \
        --softmax_op \
        --alpha_cos  ${al_COSINE} \
        --alpha_iou  ${al_IOU} \
        --alpha_kl  ${al_KL} \
        --alpha_sim  ${al_SIM} \
        --beta_cos ${be_COSINE} \
        --beta_iou ${be_IOU} \
        --beta_kl ${be_KL} \
        --beta_sim ${be_SIM} \
        --lam_cos ${lam_COSINE} \
        --lam_iou ${lam_IOU} \
        --lam_kl ${lam_KL} \
        --lam_sim ${lam_SIM} \
        --null_inner_steps ${NULL_STEP} \
        --attn_inner_steps ${ATTN_STEP} \

done


### NOTE: ablate 4
### NOTE: Ablation study parameters
al_COSINE=25.0
al_IOU=25.0  
be_COSINE=0.5 
be_IOU=0.5
lam_COSINE=0.1
lam_IOU=1.0

### NOTE: 0.5 > 0.3/0.7

for be_IOU in 0.3 0.7
do

echo "current image is ${FILE_NAME}.${IMG_FORMAT}"

CUDA_VISIBLE_DEVICES=5 \
python _3_det_null_textinv.py \
        --input_image ${IMG_FOLDER}/${OBJECT}/${SEARCH_TEXT}/${FILE_NAME}/${FILE_NAME}.${IMG_FORMAT} \
        --results_folder ${IMG_FOLDER}/${OBJECT}/${SEARCH_TEXT}/${FILE_NAME}/ \
        --seg_dirs ${IMG_FOLDER}/${OBJECT}/${SEARCH_TEXT}/${FILE_NAME}/ \
        --negative_guidance_scale 7.5 \
        --print_freq 10 \
        --placeholder_token ${PLACEHOLDER1}    \
        --initializer_token ${INIT_TOKEN1}   \
        --smooth_op \
        --softmax_op \
        --alpha_cos  ${al_COSINE} \
        --alpha_iou  ${al_IOU} \
        --alpha_kl  ${al_KL} \
        --alpha_sim  ${al_SIM} \
        --beta_cos ${be_COSINE} \
        --beta_iou ${be_IOU} \
        --beta_kl ${be_KL} \
        --beta_sim ${be_SIM} \
        --lam_cos ${lam_COSINE} \
        --lam_iou ${lam_IOU} \
        --lam_kl ${lam_KL} \
        --lam_sim ${lam_SIM} \
        --null_inner_steps ${NULL_STEP} \
        --attn_inner_steps ${ATTN_STEP} \

done



### NOTE: ablate 5
### NOTE: Ablation study parameters
al_COSINE=25.0
al_IOU=25.0  
be_COSINE=0.5 
be_IOU=0.5
lam_COSINE=0.1
lam_IOU=1.0

### NOTE:0.1 0.0 0.5 similar > 5.0

for lam_COSINE in 0.0 0.5 1.0 2.0 5.0
do

echo "current image is ${FILE_NAME}.${IMG_FORMAT}"

CUDA_VISIBLE_DEVICES=5 \
python _3_det_null_textinv.py \
        --input_image ${IMG_FOLDER}/${OBJECT}/${SEARCH_TEXT}/${FILE_NAME}/${FILE_NAME}.${IMG_FORMAT} \
        --results_folder ${IMG_FOLDER}/${OBJECT}/${SEARCH_TEXT}/${FILE_NAME}/ \
        --seg_dirs ${IMG_FOLDER}/${OBJECT}/${SEARCH_TEXT}/${FILE_NAME}/ \
        --negative_guidance_scale 7.5 \
        --print_freq 10 \
        --placeholder_token ${PLACEHOLDER1}    \
        --initializer_token ${INIT_TOKEN1}   \
        --smooth_op \
        --softmax_op \
        --alpha_cos  ${al_COSINE} \
        --alpha_iou  ${al_IOU} \
        --alpha_kl  ${al_KL} \
        --alpha_sim  ${al_SIM} \
        --beta_cos ${be_COSINE} \
        --beta_iou ${be_IOU} \
        --beta_kl ${be_KL} \
        --beta_sim ${be_SIM} \
        --lam_cos ${lam_COSINE} \
        --lam_iou ${lam_IOU} \
        --lam_kl ${lam_KL} \
        --lam_sim ${lam_SIM} \
        --null_inner_steps ${NULL_STEP} \
        --attn_inner_steps ${ATTN_STEP} \

done



### NOTE: ablate 6
### NOTE: Ablation study parameters
al_COSINE=25.0
al_IOU=25.0  
be_COSINE=0.5 
be_IOU=0.5
lam_COSINE=0.1
lam_IOU=1.0

### NOTE: worse 

for lam_IOU in 0.0
do

echo "current image is ${FILE_NAME}.${IMG_FORMAT}"

CUDA_VISIBLE_DEVICES=5 \
python _3_det_null_textinv.py \
        --input_image ${IMG_FOLDER}/${OBJECT}/${SEARCH_TEXT}/${FILE_NAME}/${FILE_NAME}.${IMG_FORMAT} \
        --results_folder ${IMG_FOLDER}/${OBJECT}/${SEARCH_TEXT}/${FILE_NAME}/ \
        --seg_dirs ${IMG_FOLDER}/${OBJECT}/${SEARCH_TEXT}/${FILE_NAME}/ \
        --negative_guidance_scale 7.5 \
        --print_freq 10 \
        --placeholder_token ${PLACEHOLDER1}    \
        --initializer_token ${INIT_TOKEN1}   \
        --smooth_op \
        --softmax_op \
        --alpha_cos  ${al_COSINE} \
        --alpha_iou  ${al_IOU} \
        --alpha_kl  ${al_KL} \
        --alpha_sim  ${al_SIM} \
        --beta_cos ${be_COSINE} \
        --beta_iou ${be_IOU} \
        --beta_kl ${be_KL} \
        --beta_sim ${be_SIM} \
        --lam_cos ${lam_COSINE} \
        --lam_iou ${lam_IOU} \
        --lam_kl ${lam_KL} \
        --lam_sim ${lam_SIM} \
        --null_inner_steps ${NULL_STEP} \
        --attn_inner_steps ${ATTN_STEP} \

done


done

### NOTE: ablate 7
### NOTE: Ablation study parameters
# al_COSINE=50.0
# al_IOU=25.0
# al_SIM=25.0

# be_COSINE=0.3
# be_IOU=0.5
# be_SIM=0.9

# lam_COSINE=0.1
# lam_IOU=1.0
# lam_SIM=1.0

# ### NOTE: not useful. should be 0.0

# for lam_SIM in 0.1 0.5 1.0 0.2 0.05 2.0
# do

# echo "current image is ${FILE_NAME}.${IMG_FORMAT}"

# CUDA_VISIBLE_DEVICES=5 \
# python _3_det_null_textinv.py \
#         --input_image ${IMG_FOLDER}/${FILE_NAME}.${IMG_FORMAT} \
#         --results_folder output/${FILE_NAME}/ \
#         --seg_dirs seg_dirs/${FILE_NAME}/ \
#         --negative_guidance_scale 7.5 \
#         --print_freq 10 \
#         --placeholder_token ${PLACEHOLDER1} ${PLACEHOLDER2}  ${PLACEHOLDER3} \
#         --initializer_token ${INIT_TOKEN1} ${INIT_TOKEN2} ${INIT_TOKEN3}  \
#         --smooth_op \
#         --softmax_op \
#         --alpha_cos  ${al_COSINE} \
#         --alpha_iou  ${al_IOU} \
#         --alpha_kl  ${al_KL} \
#         --alpha_sim  ${al_SIM} \
#         --beta_cos ${be_COSINE} \
#         --beta_iou ${be_IOU} \
#         --beta_kl ${be_KL} \
#         --beta_sim ${be_SIM} \
#         --lam_cos ${lam_COSINE} \
#         --lam_iou ${lam_IOU} \
#         --lam_kl ${lam_KL} \
#         --lam_sim ${lam_SIM} \
#         --null_inner_steps ${NULL_STEP} \
#         --attn_inner_steps ${ATTN_STEP} \

# done