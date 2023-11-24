

for object_name in 'a cartoon of' 'a portrait of' 'a watercolor of'  

do

FILE_NAME="catdog"

# IMG_FORMAT='png'
IMG_FORMAT='jpg'
INDICES='2 5'
AMPLIFY='2'
SCALE='2.0 2.0'

PROMPT="a <cat-toy> and a <dog-toy>"

EDIT_PROMPT="${object_name} a <cat-toy> and a <dog-toy> "

PLACEHOLDER1='<cat-toy>'
PLACEHOLDER2='<dog-toy>'
INIT_TOKEN1='cat'
INIT_TOKEN2='dog'

CUDA_VISIBLE_DEVICES=9 python _image_edit.py \
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
    --refine \
    --indices_to_amplify ${AMPLIFY} \
    --amplify_scale ${SCALE} \

done