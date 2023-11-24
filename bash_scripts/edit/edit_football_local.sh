
for object_name in 'apple' 'basketball' 'balloon' 'football' 'donut' 'earth' 'moon' 'sunflower' 'plate' 'wheel'

do

FILE_NAME="a_football_on_a_bench"
FILE_ID=0
PROMPT="a <football-toy> on a <bench-toy> in a park"
EDIT_PROMPT="a ${object_name} on a <bench-toy> in a park"

IMG_FORMAT='png'
INDICES='2 5'
AMPLIFY='2'
# AMPLIFY='11'
SCALE='5.0 5.0'
LOCAL_INDICES='2'

PLACEHOLDER1='<bench-toy>'
PLACEHOLDER2='<football-toy>'

INIT_TOKEN1='bench'
INIT_TOKEN2='football'

CUDA_VISIBLE_DEVICES=7 python _image_edit.py \
    --input_image "clip_retrieval/${FILE_NAME}/${FILE_ID}.${IMG_FORMAT}" \
    --prompt_str "${PROMPT}" \
    --edit_prompt_str "${EDIT_PROMPT}" \
    --results_folder "output/${FILE_NAME}_${FILE_ID}/" \
    --results_folder_edit "output/${FILE_NAME}_${FILE_ID}/${FILE_ID}_edit/" \
    --negative_guidance_scale 7.5 \
    --indices_to_alter ${INDICES} \
    --placeholder_token ${PLACEHOLDER1} ${PLACEHOLDER2} \
    --initializer_token ${INIT_TOKEN1} ${INIT_TOKEN2} \
    --postfix '1.0_0.0_0.2_27.6_27.6_25.0_0.3_2.5_0.9_16_True_True_1.0_1.0' \
    --replace \
    --indices_local ${LOCAL_INDICES} \
    --local \
    --indices_to_amplify ${AMPLIFY} \
    --amplify_scale ${SCALE} \

done