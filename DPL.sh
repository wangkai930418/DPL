### 1st step: get the background mask and inversion files for DPL
IMG_FORMAT='jpg'
IMG_FOLDER='images/'
FILE_NAME='catdog'

CUDA_VISIBLE_DEVICES=1 python _2_DDIM_inv.py \
    --input_image ${IMG_FOLDER}/${FILE_NAME}.${IMG_FORMAT} \
    --results_folder output/ \

### 2nd step: Dynamic Prompt Learning
PLACEHOLDER1='<cat-toy>'
PLACEHOLDER2='<dog-toy>'

INIT_TOKEN1='cat'
INIT_TOKEN2='dog'

MAX_ITER=0
LOSS='max'
ATTN_RES=16

ATTN=1.0
BG=0.05
COSINE=0.05

NULL_STEP=51
ATTN_STEP=21

al_ATTN=25.0 
al_BG=50.0
al_COSINE=25.0

be_ATTN=0.3
be_BG=0.7
be_COSINE=0.9

CUDA_VISIBLE_DEVICES=0 python _3_dpl_inv.py \
    --input_image ${IMG_FOLDER}/${FILE_NAME}.${IMG_FORMAT} \
    --results_folder output/${FILE_NAME}/ \
    --negative_guidance_scale 7.5 \
    --null_inner_steps ${NULL_STEP} \
    --attn_inner_steps ${ATTN_STEP} \
    --lam_maxattn ${ATTN} \
    --lam_entropy ${BG} \
    --lam_cosine ${COSINE} \
    --print_freq 10 \
    --max_iter_to_alter ${MAX_ITER} \
    --placeholder_token ${PLACEHOLDER1} ${PLACEHOLDER2}   \
    --initializer_token ${INIT_TOKEN1} ${INIT_TOKEN2}   \
    --alpha_max  ${al_ATTN} \
    --alpha_ent  ${al_BG} \
    --alpha_cos  ${al_COSINE} \
    --beta_max  ${be_ATTN} \
    --beta_ent  ${be_BG} \
    --beta_cos  ${be_COSINE} \
    --loss_type ${LOSS} \
    --attn_res ${ATTN_RES} \
    --smooth_op \
    --softmax_op \
    --seg_dirs output/${FILE_NAME}/sd_study/ \
