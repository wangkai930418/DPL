
for object_name in 'zebra' 'unicorn' 'mule' 'buffalo' 'cow' 'cattle'

do

FILE_NAME="a_horse_and_a_sheep"
FILE_ID=190
PROMPT="a <horse-toy> and a <sheep-toy> in a field"
EDIT_PROMPT="a ${object_name} and a <sheep-toy> in a field"

IMG_FORMAT='png'
INDICES='2 5'
AMPLIFY='2'
SCALE='5.0 5.0'
LOCAL_INDICES='2'

PLACEHOLDER1='<horse-toy>'
PLACEHOLDER2='<dog-toy>'
PLACEHOLDER3='<girl-toy>'
PLACEHOLDER4='<sheep-toy>'

INIT_TOKEN1='horse'
INIT_TOKEN2='dog'
INIT_TOKEN3='girl'
INIT_TOKEN4='sheep'

CUDA_VISIBLE_DEVICES=7 python _image_edit.py \
    --input_image "clip_retrieval/${FILE_NAME}/${FILE_ID}.${IMG_FORMAT}" \
    --prompt_str "${PROMPT}" \
    --edit_prompt_str "${EDIT_PROMPT}" \
    --results_folder "output/${FILE_NAME}_${FILE_ID}/" \
    --results_folder_edit "output/${FILE_NAME}_${FILE_ID}/${FILE_ID}_edit/" \
    --negative_guidance_scale 7.5 \
    --indices_to_alter ${INDICES} \
    --placeholder_token ${PLACEHOLDER1} ${PLACEHOLDER2} ${PLACEHOLDER3} ${PLACEHOLDER4} \
    --initializer_token ${INIT_TOKEN1} ${INIT_TOKEN2} ${INIT_TOKEN3} ${INIT_TOKEN4} \
    --postfix '1.0_0.0_0.2_27.6_27.6_25.0_0.3_2.5_0.9_16_True_True_1.0_1.0' \
    --replace \
    --indices_to_amplify ${AMPLIFY} \
    --amplify_scale ${SCALE} \
    # --indices_local ${LOCAL_INDICES} \
    # --local \

done