for object_name in 'pineapple' 'grape' 'mango' 'limon' 'peach' 'strawberry'

do

FILE_NAME="apple_orange_pepper"
PROMPT="a <apple-toy> a <orange-toy> and a <pepper-toy>"
EDIT_PROMPT="a ${object_name} a <orange-toy> and a <pepper-toy>"

INDICES='2 4 7'
PLACEHOLDER1='<apple-toy>'
PLACEHOLDER2='<orange-toy>'
PLACEHOLDER3='<pepper-toy>'
INIT_TOKEN1='apple'
INIT_TOKEN2='orange'
INIT_TOKEN3='pepper'

CUDA_VISIBLE_DEVICES=0 python image_edit.py \
    --input_image "images/${FILE_NAME}.jpg" \
    --prompt_str "${PROMPT}" \
    --edit_prompt_str "${EDIT_PROMPT}" \
    --results_folder "output/${FILE_NAME}/" \
    --results_folder_edit "output/${FILE_NAME}_edit/" \
    --negative_guidance_scale 7.5 \
    --indices_to_alter ${INDICES} \
    --placeholder_token ${PLACEHOLDER1} ${PLACEHOLDER2} ${PLACEHOLDER3} \
    --initializer_token ${INIT_TOKEN1} ${INIT_TOKEN2} ${INIT_TOKEN3} \
    --postfix '1.0_0.0_0.2_27.6_27.6_25.0_0.3_2.5_0.9_16_True_True_1.0_1.0' \


EDIT_PROMPT="a <apple-toy> a ${object_name} and a <pepper-toy>"
CUDA_VISIBLE_DEVICES=0 python image_edit.py \
    --input_image "images/${FILE_NAME}.jpg" \
    --prompt_str "${PROMPT}" \
    --edit_prompt_str "${EDIT_PROMPT}" \
    --results_folder "output/${FILE_NAME}/" \
    --results_folder_edit "output/${FILE_NAME}_edit/" \
    --negative_guidance_scale 7.5 \
    --indices_to_alter ${INDICES} \
    --placeholder_token ${PLACEHOLDER1} ${PLACEHOLDER2} ${PLACEHOLDER3} \
    --initializer_token ${INIT_TOKEN1} ${INIT_TOKEN2} ${INIT_TOKEN3} \
    --postfix '1.0_0.0_0.2_27.6_27.6_25.0_0.3_2.5_0.9_16_True_True_1.0_1.0' \


EDIT_PROMPT="a <apple-toy> a <orange-toy> and a ${object_name}"
CUDA_VISIBLE_DEVICES=0 python image_edit.py \
    --input_image "images/${FILE_NAME}.jpg" \
    --prompt_str "${PROMPT}" \
    --edit_prompt_str "${EDIT_PROMPT}" \
    --results_folder "output/${FILE_NAME}/" \
    --results_folder_edit "output/${FILE_NAME}_edit/" \
    --negative_guidance_scale 7.5 \
    --indices_to_alter ${INDICES} \
    --placeholder_token ${PLACEHOLDER1} ${PLACEHOLDER2} ${PLACEHOLDER3} \
    --initializer_token ${INIT_TOKEN1} ${INIT_TOKEN2} ${INIT_TOKEN3} \
    --postfix '1.0_0.0_0.2_27.6_27.6_25.0_0.3_2.5_0.9_16_True_True_1.0_1.0' \


done