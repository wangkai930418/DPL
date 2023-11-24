
for object_name in 'fox' 'tiger' 'lion' 'mouse' 'koala' 'gorilla' 'raccon' 'zebra' 'wolf' 'deer' 'horse' 'rabbit' 'leopard' 'monkey'

do

FILE_NAME="lexica_dog_cat"

IMG_FORMAT='jpg'
INDICES='2 5'
LOCAL_INDICES='2'
POSTFIX='local_edit'

PROMPT="a <dog-toy> and a <cat-toy> sitting on a gray background"
EDIT_PROMPT="a ${object_name} and a <cat-toy> sitting on a gray background"

PLACEHOLDER1='<dog-toy>'
PLACEHOLDER2='<cat-toy>'
INIT_TOKEN1='dog'
INIT_TOKEN2='cat'

CUDA_VISIBLE_DEVICES=6 python _image_edit.py \
    --input_image "images/${FILE_NAME}.${IMG_FORMAT}" \
    --prompt_str "${PROMPT}" \
    --edit_prompt_str "${EDIT_PROMPT}" \
    --results_folder "output/${FILE_NAME}/" \
    --results_folder_edit "output/${FILE_NAME}_${POSTFIX}/" \
    --negative_guidance_scale 7.5 \
    --indices_to_alter  ${INDICES} \
    --placeholder_token ${PLACEHOLDER1} ${PLACEHOLDER2} \
    --initializer_token ${INIT_TOKEN1} ${INIT_TOKEN2} \
    --postfix '1.0_0.0_0.2_27.6_27.6_25.0_0.3_2.5_0.9_16_True_True_1.0_1.0' \
    --replace \
    --local \
    --indices_local ${LOCAL_INDICES}

EDIT_PROMPT="a <dog-toy> and a ${object_name} sitting on a gray background"
LOCAL_INDICES='5'

CUDA_VISIBLE_DEVICES=6 python _image_edit.py \
    --input_image "images/${FILE_NAME}.${IMG_FORMAT}" \
    --prompt_str "${PROMPT}" \
    --edit_prompt_str "${EDIT_PROMPT}" \
    --results_folder "output/${FILE_NAME}/" \
    --results_folder_edit "output/${FILE_NAME}_${POSTFIX}/" \
    --negative_guidance_scale 7.5 \
    --indices_to_alter  ${INDICES} \
    --placeholder_token ${PLACEHOLDER1} ${PLACEHOLDER2} \
    --initializer_token ${INIT_TOKEN1} ${INIT_TOKEN2} \
    --postfix '1.0_0.0_0.2_27.6_27.6_25.0_0.3_2.5_0.9_16_True_True_1.0_1.0' \
    --replace \
    --local \
    --indices_local ${LOCAL_INDICES}

done