
for object_name in 'fox' 'tiger' 'lion' 'mouse' 'koala' 'gorilla' 'raccon' 'zebra' 'wolf' 'deer' 'horse' 'rabbit' 'leopard' 'monkey'
do

FILE_NAME="cat_ball"
IMG_FORMAT='jpg'
INDICES='3 8'

PROMPT="a small <cat-toy> playing with a pink <ball-toy>"
EDIT_PROMPT="a small ${object_name} playing with a pink <ball-toy>"

PLACEHOLDER1='<cat-toy>'
PLACEHOLDER2='<ball-toy>'
INIT_TOKEN1='cat'
INIT_TOKEN2='ball'

CUDA_VISIBLE_DEVICES=1 python image_edit.py \
    --input_image "images/${FILE_NAME}.${IMG_FORMAT}" \
    --prompt_str "${PROMPT}" \
    --edit_prompt_str "${EDIT_PROMPT}" \
    --results_folder "output/${FILE_NAME}/" \
    --results_folder_edit "output/${FILE_NAME}_edit/" \
    --negative_guidance_scale 7.5 \
    --indices_to_alter  ${INDICES} \
    --placeholder_token ${PLACEHOLDER1} ${PLACEHOLDER2} \
    --initializer_token ${INIT_TOKEN1} ${INIT_TOKEN2} \
    --postfix '1.0_0.0_0.2_27.6_27.6_25.0_0.3_2.5_0.9_16_True_True_1.0_1.0' \

done


for object_name in 'fish' 'pufferfish' 'basketball' 'football' 'pingpong' 
do

FILE_NAME="cat_ball"
IMG_FORMAT='jpg'
INDICES='3 8'

PROMPT="a small <cat-toy> playing with a pink <ball-toy>"
EDIT_PROMPT="a small <cat-toy> playing with a pink ${object_name}"

PLACEHOLDER1='<cat-toy>'
PLACEHOLDER2='<ball-toy>'
INIT_TOKEN1='cat'
INIT_TOKEN2='ball'

CUDA_VISIBLE_DEVICES=1 python image_edit.py \
    --input_image "images/${FILE_NAME}.${IMG_FORMAT}" \
    --prompt_str "${PROMPT}" \
    --edit_prompt_str "${EDIT_PROMPT}" \
    --results_folder "output/${FILE_NAME}/" \
    --results_folder_edit "output/${FILE_NAME}_edit/" \
    --negative_guidance_scale 7.5 \
    --indices_to_alter  ${INDICES} \
    --placeholder_token ${PLACEHOLDER1} ${PLACEHOLDER2} \
    --initializer_token ${INIT_TOKEN1} ${INIT_TOKEN2} \
    --postfix '1.0_0.0_0.2_27.6_27.6_25.0_0.3_2.5_0.9_16_True_True_1.0_1.0' \

done
