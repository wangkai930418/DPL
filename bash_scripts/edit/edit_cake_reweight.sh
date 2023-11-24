
# object_name1='box'
# object_name2='sunflower'

IMG_FORMAT='png'

IMG_FOLDER='clip_retrieval/'
SEARCH_TEXT='a_cake_in_a_plate'


POSTFIX='1.0_0.0_0.2_27.6_27.6_25.0_0.3_2.5_0.9_16_True_True_1.0_1.0'
POSTFIX='0.0_0.0_0.0_27.6_27.6_25.0_0.3_2.5_0.9_16_True_True_1.0_1.0'
AMPLIFY='4'
# LOCAL='4'
SCALE='1.0 2.0'


PLACEHOLDER1='<cake-toy>'
PLACEHOLDER2='<plate-toy>'

INIT_TOKEN1='cake'
INIT_TOKEN2='plate'


for FILE_NAME in 21
# for FILE_NAME in 237
do

for object_name1 in 'lemon' 'apple' 'beet' 'pasta'  'brick' 'chocolate'   
# for object_name1 in 'wooden'  'metal'  'bamboo' 'plastic' 'green' 'orange' 'yellow' 'blue' 'pink'
# for object_name1 in 'green' 'orange' 'yellow' 'blue' 'pink'
do

# EDIT_PROMPT="a white <plate-toy> with a piece of ${object_name1} <cake-toy> on it" ### 237
# EDIT_PROMPT="a white <plate-toy> with a piece of ${object_name1} <cake-toy> on it" ### 237
EDIT_PROMPT="a piece of ${object_name1} <cake-toy> on a wooden <plate-toy>" ### 21
# a white plate with a piece of cake on it

CUDA_VISIBLE_DEVICES=1 python _4_image_edit.py \
    --input_image ${IMG_FOLDER}/${SEARCH_TEXT}/${FILE_NAME}.${IMG_FORMAT} \
    --results_folder output/${SEARCH_TEXT}_${FILE_NAME}/ \
    --results_folder_edit output/${SEARCH_TEXT}_${FILE_NAME}/${FILE_NAME}_edit \
    --negative_guidance_scale 7.5 \
    --placeholder_token ${PLACEHOLDER1} ${PLACEHOLDER2} \
    --initializer_token ${INIT_TOKEN1} ${INIT_TOKEN2} \
    --postfix ${POSTFIX} \
    --edit_prompt_str "${EDIT_PROMPT}" \
    --refine \
    --indices_to_amplify ${AMPLIFY} \
    --amplify_scale ${SCALE} \
    --adj_indice ${AMPLIFY} \
    # --indices_local ${LOCAL} \
    # --local
    # --replace \
    # --original_words  '<clock-toy>' \
    # --replace_words  ${object_name2}
    # --original_words '<book-toy>' '<books-toy>' '<clock-toy>' \
    # --replace_words ${object_name1} ${object_name1} ${object_name2}

done

done
