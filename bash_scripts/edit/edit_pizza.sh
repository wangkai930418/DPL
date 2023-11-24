IMG_FORMAT='png'

IMG_FOLDER='clip_retrieval/'
SEARCH_TEXT='a_pizza_on_a_table'

POSTFIX='1.0_0.0_0.2_27.6_27.6_25.0_0.3_2.5_0.9_16_True_True_1.0_1.0'
# POSTFIX='0.0_0.0_0.0_27.6_27.6_25.0_0.3_2.5_0.9_16_True_True_1.0_1.0'
AMPLIFY='2'
SCALE='1.0 2.0'


PLACEHOLDER1='<pizza-toy>'
PLACEHOLDER2='<table-toy>'
INIT_TOKEN1='pizza'
INIT_TOKEN2='table'


# for FILE_NAME in 93 
for FILE_NAME in 27
do
# 
# for object_name1 in 'lemon'  'pasta'  'pineapple'  'mango' 'apple' 'jamon' 'dumplings' 
for object_name1 in 'lemon'  'pasta'  'dumplings'  'pineapple' 
do

EDIT_PROMPT="a ${object_name1} <pizza-toy> on a wooden <table-toy>" ### 93 27

echo ${EDIT_PROMPT}

CUDA_VISIBLE_DEVICES=1 python _4_image_edit.py \
    --input_image ${IMG_FOLDER}/${SEARCH_TEXT}/${FILE_NAME}.${IMG_FORMAT} \
    --results_folder output/${SEARCH_TEXT}_${FILE_NAME}/ \
    --results_folder_edit output/${SEARCH_TEXT}_${FILE_NAME}/${FILE_NAME}_edit \
    --negative_guidance_scale 7.5 \
    --placeholder_token ${PLACEHOLDER1} ${PLACEHOLDER2} \
    --initializer_token ${INIT_TOKEN1} ${INIT_TOKEN2} \
    --postfix ${POSTFIX} \
    --edit_prompt_str="${EDIT_PROMPT}" \
    --refine \
    --indices_to_amplify ${AMPLIFY} \
    --amplify_scale ${SCALE} \
    --adj_indice 2 \
    # --replace \
    # --original_words  '<clock-toy>' \
    # --replace_words  ${object_name2}
    # --original_words '<book-toy>' '<books-toy>' '<clock-toy>' \
    # --replace_words ${object_name1} ${object_name1} ${object_name2}


done

done
