for object_name in 'pineapple' 'grape' 'mango' 'limon' 'peach' 'strawberry'

do

FILE_NAME="pear_apple"
PROMPT="a <pear-toy> and a <apple-toy>"
EDIT_PROMPT="a ${object_name} and a <apple-toy>"

PLACEHOLDER1='<pear-toy>'
PLACEHOLDER2='<apple-toy>'
INIT_TOKEN1='pear'
INIT_TOKEN2='apple'

CUDA_VISIBLE_DEVICES=0 python image_edit.py \
    --input_image "images/${FILE_NAME}.jpg" \
    --prompt_str "${PROMPT}" \
    --edit_prompt_str "${EDIT_PROMPT}" \
    --results_folder "output/${FILE_NAME}/" \
    --results_folder_edit "output/${FILE_NAME}_edit/" \
    --negative_guidance_scale 7.5 \
    --indices_to_alter 2 5 \
    --placeholder_token ${PLACEHOLDER1} ${PLACEHOLDER2} \
    --initializer_token ${INIT_TOKEN1} ${INIT_TOKEN2} \
    --postfix '1.0_0.0_0.2_27.6_27.6_25.0_0.3_2.5_0.9_16_True_True_1.0_1.0' \


EDIT_PROMPT="a <pear-toy> and a ${object_name}"
CUDA_VISIBLE_DEVICES=0 python image_edit.py \
    --input_image "images/${FILE_NAME}.jpg" \
    --prompt_str "${PROMPT}" \
    --edit_prompt_str "${EDIT_PROMPT}" \
    --results_folder "output/${FILE_NAME}/" \
    --results_folder_edit "output/${FILE_NAME}_edit/" \
    --negative_guidance_scale 7.5 \
    --indices_to_alter 2 5 \
    --placeholder_token ${PLACEHOLDER1} ${PLACEHOLDER2} \
    --initializer_token ${INIT_TOKEN1} ${INIT_TOKEN2} \
    --postfix '1.0_0.0_0.2_27.6_27.6_25.0_0.3_2.5_0.9_16_True_True_1.0_1.0' \


done