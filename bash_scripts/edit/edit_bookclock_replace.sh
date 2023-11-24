
object_name1='box'

IMG_FORMAT='png'

IMG_FOLDER='clip_retrieval/'
SEARCH_TEXT='a_book_and_a_clock'
# POSTFIX='1.0_0.0_0.2_27.6_27.6_25.0_0.3_2.5_0.9_16_True_True_1.0_1.0'
POSTFIX='0.0_0.0_0.0_27.6_27.6_25.0_0.3_2.5_0.9_16_True_True_1.0_1.0'
FILE_NAME=1

PLACEHOLDER1='<book-toy>'
PLACEHOLDER2='<clock-toy>'
PLACEHOLDER3='<books-toy>'
PLACEHOLDER4='<table-toy>'

INIT_TOKEN1='book'
INIT_TOKEN2='clock'
INIT_TOKEN3='books'
INIT_TOKEN4='table'

object_name2='sunflower'

for object_name2 in  'wheel' 'sunflower'  'donut' 'coin'
do

CUDA_VISIBLE_DEVICES=1 python _4_image_edit.py \
    --input_image ${IMG_FOLDER}/${SEARCH_TEXT}/${FILE_NAME}.${IMG_FORMAT} \
    --results_folder output/${SEARCH_TEXT}_${FILE_NAME}/ \
    --results_folder_edit output/${SEARCH_TEXT}_${FILE_NAME}/${FILE_NAME}_edit \
    --negative_guidance_scale 7.5 \
    --placeholder_token ${PLACEHOLDER1} ${PLACEHOLDER2} ${PLACEHOLDER3} ${PLACEHOLDER4} \
    --initializer_token ${INIT_TOKEN1} ${INIT_TOKEN2} ${INIT_TOKEN3} ${INIT_TOKEN4} \
    --postfix ${POSTFIX} \
    --replace \
    --original_words  '<clock-toy>' \
    --replace_words  ${object_name2}

done


