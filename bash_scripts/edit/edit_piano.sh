
# object_name1='box'
# object_name2='sunflower'

IMG_FORMAT='png'

IMG_FOLDER='clip_retrieval/'
SEARCH_TEXT='a_piano_and_a_chair_in_the_room'


POSTFIX='1.0_0.0_0.2_27.6_27.6_25.0_0.3_2.5_0.9_16_True_True_1.0_1.0'
# POSTFIX='0.0_0.0_0.0_27.6_27.6_25.0_0.3_2.5_0.9_16_True_True_1.0_1.0'
POSTFIX='0.0_0.0_0.0_25.0_50.0_25.0_0.3_0.7_0.9_0'

AMPLIFY='8'
ADJ='2'
# LOCAL='4'
SCALE='1.0 2.0'

PLACEHOLDER1='<piano-toy>'
PLACEHOLDER2='<chair-toy>'
INIT_TOKEN1='piano'
INIT_TOKEN2='chair'


for FILE_NAME in 131
do

for object_name1 in  'box' # 'brick' 'window'  'book'
# for object_name1 in 'sofa' 'box' 'table' 'stool' 'armchair' 'bench'
# for object_name1 in 'book' 'television' 'phone' 'window' 'screen' 'whiteboard' 'brick'

do

# EDIT_PROMPT="a white <plate-toy> with a piece of ${object_name1} <cake-toy> on it" ### 237
# EDIT_PROMPT="a white <plate-toy> with a piece of ${object_name1} <cake-toy> on it" ### 237
# EDIT_PROMPT="a piece of ${object_name1} <cake-toy> on a wooden <plate-toy>" ### 21
# a white plate with a piece of cake on it

CUDA_VISIBLE_DEVICES=1 python _4_image_edit.py \
    --input_image ${IMG_FOLDER}/${SEARCH_TEXT}/${FILE_NAME}.${IMG_FORMAT} \
    --results_folder output/${SEARCH_TEXT}_${FILE_NAME}/ \
    --results_folder_edit output/${SEARCH_TEXT}_${FILE_NAME}/${FILE_NAME}_edit \
    --negative_guidance_scale 7.5 \
    --placeholder_token ${PLACEHOLDER1} ${PLACEHOLDER2} \
    --initializer_token ${INIT_TOKEN1} ${INIT_TOKEN2} \
    --postfix ${POSTFIX} \
    --replace \
    --original_words '<chair-toy>' \
    --replace_words ${object_name1} \
    # --adj_indice ${ADJ} \

done

done
