
# for object_name in 'fox' 'tiger' 'lion' 'mouse' 'koala' 'gorilla' 'raccon' 'zebra' 'wolf' 'deer' 'horse' 'rabbit' 'leopard' 'monkey'
# for object_name in 'fish' 'chicken' 'bird' 'frog' 'snake' 'dragon' 'crab' 'shark' 'shell'

for CROSS_STEP in  6 7  8 9 11 12 13 14 16 17 18 19 ### 0 5 10 15 20 25 30 40 50
do

for object_name in 'wolf' # 'lion' 'zebra' 

do

FILE_NAME="catdog"

# SELF_STEP=20

# IMG_FORMAT='png'
IMG_FORMAT='jpg'
INDICES='2 5'
LOCAL_INDICES='2'

PROMPT="a <cat-toy> and a <dog-toy>"
EDIT_PROMPT="a ${object_name} and a <dog-toy>"

PLACEHOLDER1='<cat-toy>'
PLACEHOLDER2='<dog-toy>'
INIT_TOKEN1='cat'
INIT_TOKEN2='dog'
# /home/kwang/.cache/huggingface/diffusers/models--CompVis--stable-diffusion-v1-4/snapshots/b95be7d6f134c3a9e62ee616f310733567f069ce

CUDA_VISIBLE_DEVICES=1 python _4_image_edit.py \
    --input_image "images/${FILE_NAME}.${IMG_FORMAT}" \
    --prompt_str "${PROMPT}" \
    --edit_prompt_str "${EDIT_PROMPT}" \
    --results_folder "output/${FILE_NAME}/" \
    --results_folder_edit "output/${FILE_NAME}_edit_gradual/" \
    --negative_guidance_scale 7.5 \
    --indices_to_alter  ${INDICES} \
    --placeholder_token ${PLACEHOLDER1} ${PLACEHOLDER2} \
    --initializer_token ${INIT_TOKEN1} ${INIT_TOKEN2} \
    --postfix '1.0_0.05_0.05_25.0_50.0_25.0_0.3_0.8_0.9_0' \
    --replace \
    --cross_replace_steps ${CROSS_STEP} \
    --self_replace_steps ${CROSS_STEP} \
    # --local \
    # --indices_local ${LOCAL_INDICES}

# EDIT_PROMPT="a <cat-toy> and a ${object_name}"
# LOCAL_INDICES='5'

# CUDA_VISIBLE_DEVICES=5 python _image_edit.py \
#     --input_image "images/${FILE_NAME}.${IMG_FORMAT}" \
#     --prompt_str "${PROMPT}" \
#     --edit_prompt_str "${EDIT_PROMPT}" \
#     --results_folder "output/${FILE_NAME}/" \
#     --results_folder_edit "output/${FILE_NAME}_edit/" \
#     --negative_guidance_scale 7.5 \
#     --indices_to_alter  ${INDICES} \
#     --placeholder_token ${PLACEHOLDER1} ${PLACEHOLDER2} \
#     --initializer_token ${INIT_TOKEN1} ${INIT_TOKEN2} \
#     --postfix '1.0_0.0_0.2_27.6_27.6_25.0_0.3_2.5_0.9_16_True_True_1.0_1.0' \
#     --replace \
#     --local \
#     --indices_local ${LOCAL_INDICES}

done
done