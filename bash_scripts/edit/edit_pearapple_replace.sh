IMG_FORMAT='jpg'

IMG_FOLDER='images/'
SEARCH_TEXT='pear_apple'


POSTFIX='1.0_0.0_0.2_27.6_27.6_25.0_0.3_2.5_0.9_16_True_True_1.0_1.0'
# POSTFIX='0.0_0.0_0.0_25.0_50.0_25.0_0.3_0.7_0.9_0'
# AMPLIFY='8'
# ADJ='2'
# LOCAL='4'
# SCALE='1.0 1.0'


PLACEHOLDER1='<pear-toy>'
PLACEHOLDER2='<apple-toy>'
INIT_TOKEN1='pear'
INIT_TOKEN2='apple'

# for FILE_NAME in 21
for FILE_NAME in 'pear_apple'
do

for CROSS_STEP in 10 20 30 40 50  
do

for object_name1 in  'strawberry' ### 'orange'   'pineapple'  'strawberry' 'watermelon'
do

CUDA_VISIBLE_DEVICES=1 python _4_image_edit.py \
    --input_image ${IMG_FOLDER}/${FILE_NAME}.${IMG_FORMAT} \
    --results_folder output/${FILE_NAME}/ \
    --results_folder_edit output/${FILE_NAME}_edit \
    --negative_guidance_scale 7.5 \
    --placeholder_token ${PLACEHOLDER1} ${PLACEHOLDER2} \
    --initializer_token ${INIT_TOKEN1} ${INIT_TOKEN2} \
    --postfix ${POSTFIX} \
    --replace \
    --original_words '<pear-toy>' \
    --replace_words ${object_name1} \
    --cross_replace_steps ${CROSS_STEP} \
    # --self_replace_steps ${CROSS_STEP} \

# CUDA_VISIBLE_DEVICES=1 python _4_image_edit.py \
#     --input_image ${IMG_FOLDER}/${FILE_NAME}.${IMG_FORMAT} \
#     --results_folder output/${FILE_NAME}/ \
#     --results_folder_edit output/${FILE_NAME}_edit \
#     --negative_guidance_scale 7.5 \
#     --placeholder_token ${PLACEHOLDER1} ${PLACEHOLDER2} \
#     --initializer_token ${INIT_TOKEN1} ${INIT_TOKEN2} \
#     --postfix ${POSTFIX} \
#     --replace \
#     --original_words '<apple-toy>' \
#     --replace_words ${object_name1} \

done

done
done
