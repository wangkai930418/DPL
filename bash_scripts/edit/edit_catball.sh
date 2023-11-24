
object_name='leopard'

IMG_FORMAT='png'

INDICES='2 5'
LOCAL_INDICES='2'
IMG_FOLDER='clip_retrieval/'

SEARCH_TEXT='a_cat_and_a_ball'
# POSTFIX='1.0_0.0_0.2_27.6_27.6_25.0_0.3_2.5_0.9_16_True_True_1.0_1.0'
POSTFIX='0.0_0.0_0.0_27.6_27.6_25.0_0.3_2.5_0.9_16_True_True_1.0_1.0'

PLACEHOLDER1='<cat-toy>'
PLACEHOLDER2='<kitten-toy>'
PLACEHOLDER3='<ball-toy>'

INIT_TOKEN1='cat'
INIT_TOKEN2='kitten'
INIT_TOKEN3='ball'

for FILE_NAME in 0 103 107 138 14 162 170 174 203 26 46 48 52 60 67 7 72 73 80
do

CUDA_VISIBLE_DEVICES=9 python _image_edit.py \
    --input_image ${IMG_FOLDER}/${SEARCH_TEXT}/${FILE_NAME}.${IMG_FORMAT} \
    --results_folder output/${SEARCH_TEXT}_${FILE_NAME}/ \
    --results_folder_edit output/${SEARCH_TEXT}_${FILE_NAME}/${FILE_NAME}_edit \
    --negative_guidance_scale 7.5 \
    --placeholder_token ${PLACEHOLDER1} ${PLACEHOLDER2} ${PLACEHOLDER3} \
    --initializer_token ${INIT_TOKEN1} ${INIT_TOKEN2} ${INIT_TOKEN3} \
    --postfix ${POSTFIX} \
    --replace \
    --original_words '<cat-toy>' '<kitten-toy>' \
    --replace_words ${object_name} ${object_name} \

done



