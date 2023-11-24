
object_name1='brown'
object_name2='white'

IMG_FORMAT='png'

IMG_FOLDER='clip_retrieval/'
SEARCH_TEXT='a_cat_and_a_dog'
POSTFIX='1.0_0.0_0.2_27.6_27.6_25.0_0.3_2.5_0.9_16_True_True_1.0_1.0'
# POSTFIX='0.0_0.0_0.0_27.6_27.6_25.0_0.3_2.5_0.9_16_True_True_1.0_1.0'

AMPLIFY='2'
SCALE='1.0 2.0'

PROMPT="a <dog-toy> and a <cat-toy> are sitting on a basket of flowers"
EDIT_PROMPT="a ${object_name1} <dog-toy> and a <cat-toy> are sitting on a basket of flowers"

PLACEHOLDER1='<cat-toy>'
PLACEHOLDER2='<dog-toy>'
INIT_TOKEN1='cat'
INIT_TOKEN2='dog'

for FILE_NAME in 21
do

CUDA_VISIBLE_DEVICES=1 python _4_image_edit.py \
    --input_image ${IMG_FOLDER}/${SEARCH_TEXT}/${FILE_NAME}.${IMG_FORMAT} \
    --results_folder output/${SEARCH_TEXT}_${FILE_NAME}/ \
    --results_folder_edit output/${SEARCH_TEXT}_${FILE_NAME}/${FILE_NAME}_edit \
    --negative_guidance_scale 7.5 \
    --placeholder_token ${PLACEHOLDER1} ${PLACEHOLDER2} \
    --initializer_token ${INIT_TOKEN1} ${INIT_TOKEN2} \
    --postfix ${POSTFIX} \
    --refine \
    --prompt_str "${PROMPT}" \
    --edit_prompt_str "${EDIT_PROMPT}" \
    --indices_to_amplify ${AMPLIFY} \
    --amplify_scale ${SCALE} \
    --adj_indice ${AMPLIFY} \
    # --original_words '<cat-toy>' '<dog-toy>' \
    # --replace_words ${object_name1} ${object_name2}
    # --replace \
    # --local \
    # --indices_local 3 \

done


