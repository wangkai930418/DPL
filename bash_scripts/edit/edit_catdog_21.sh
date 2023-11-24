
# object_name1='box'
# object_name2='sunflower'

IMG_FORMAT='png'

IMG_FOLDER='clip_retrieval/'
SEARCH_TEXT='a_cat_and_a_dog'


POSTFIX='1.0_0.0_0.2_27.6_27.6_25.0_0.3_2.5_0.9_16_True_True_1.0_1.0'
# POSTFIX='0.0_0.0_0.0_27.6_27.6_25.0_0.3_2.5_0.9_16_True_True_1.0_1.0'
AMPLIFY='8'
SCALE='1.0 2.0'


PLACEHOLDER1='<cat-toy>'
PLACEHOLDER2='<dog-toy>'

INIT_TOKEN1='cat'
INIT_TOKEN2='dog'

for FILE_NAME in 21 
do


for object_name1 in 'tiger' 'lion' 'koala' 'gorilla' 'zebra' \
                     'leopard'  'fox' 'tiger' 'lion' 'mouse' \
                      'koala' 'gorilla' 'raccon' 'zebra' 'wolf' \
                       'deer' 'horse' 'rabbit' 'leopard' 'monkey'
do

for object_name2 in 'grass' 'trees' 'bubbles' 'sunflowers' 'roses'
do


EDIT_PROMPT="a <dog-toy> and a ${object_name1} are sitting on a basket of ${object_name2}"
# EDIT_PROMPT="a <dog-toy> and a ${object_name1} are sitting on a basket of flowers"
# EDIT_PROMPT="a ${object_name1} and a <cat-toy> are sitting on a basket of flowers"
# EDIT_PROMPT="a piece of ${object_name1} <cake-toy> on a wooden <plate-toy>"


CUDA_VISIBLE_DEVICES=1 python _4_image_edit.py \
    --input_image ${IMG_FOLDER}/${SEARCH_TEXT}/${FILE_NAME}.${IMG_FORMAT} \
    --results_folder output/${SEARCH_TEXT}_${FILE_NAME}/ \
    --results_folder_edit output/${SEARCH_TEXT}_${FILE_NAME}/${FILE_NAME}_edit \
    --negative_guidance_scale 7.5 \
    --placeholder_token ${PLACEHOLDER1} ${PLACEHOLDER2} \
    --initializer_token ${INIT_TOKEN1} ${INIT_TOKEN2} \
    --postfix ${POSTFIX} \
    --edit_prompt_str "${EDIT_PROMPT}" \
    --replace \
    # --refine \
    # --indices_to_amplify ${AMPLIFY} \
    # --amplify_scale ${SCALE} \
    # --adj_indice ${AMPLIFY} \
    # --original_words  '<clock-toy>' \
    # --replace_words  ${object_name2}
    # --original_words '<book-toy>' '<books-toy>' '<clock-toy>' \
    # --replace_words ${object_name1} ${object_name1} ${object_name2}

done
done
done
