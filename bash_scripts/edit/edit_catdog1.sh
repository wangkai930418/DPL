
object_name1='leopard'
object_name2='tiger'

IMG_FORMAT='png'

# INDICES='2 5'
# LOCAL_INDICES='2'
IMG_FOLDER='clip_retrieval/'
SEARCH_TEXT='a_cat_and_a_dog'
# POSTFIX='1.0_0.0_0.2_27.6_27.6_25.0_0.3_2.5_0.9_16_True_True_1.0_1.0'
POSTFIX='0.0_0.0_0.0_27.6_27.6_25.0_0.3_2.5_0.9_16_True_True_1.0_1.0'

# PROMPT="a <cat-toy> and a <dog-toy>"
# EDIT_PROMPT="a ${object_name} and a <dog-toy>"

PLACEHOLDER1='<cat-toy>'
PLACEHOLDER2='<dog-toy>'
INIT_TOKEN1='cat'
INIT_TOKEN2='dog'

for FILE_NAME in 0 1 2 3 6 9 21 42 50 51 54 55 58 61 63 64
do

CUDA_VISIBLE_DEVICES=7 python _image_edit.py \
    --input_image ${IMG_FOLDER}/${SEARCH_TEXT}/${FILE_NAME}.${IMG_FORMAT} \
    --results_folder output/${SEARCH_TEXT}_${FILE_NAME}/ \
    --results_folder_edit output/${SEARCH_TEXT}_${FILE_NAME}/${FILE_NAME}_edit \
    --negative_guidance_scale 7.5 \
    --placeholder_token ${PLACEHOLDER1} ${PLACEHOLDER2} \
    --initializer_token ${INIT_TOKEN1} ${INIT_TOKEN2} \
    --postfix ${POSTFIX} \
    --replace \
    --original_words '<cat-toy>' '<dog-toy>' \
    --replace_words ${object_name1} ${object_name2}

done


