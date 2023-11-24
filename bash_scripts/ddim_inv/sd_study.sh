
IMG_FORMAT='png'

# IMG_FOLDER='clip_retrieval/'
IMG_FOLDER='filter_clip_retrieval/'
# SEARCH_TEXT='a_cat'
# SEARCH_TEXT='a_cat_in_a_background'
SEARCH_TEXT='a_cat_and_a_dog'

PROMPT='cat'

FILE_LIST=$(ls -1 clip_retrieval/${SEARCH_TEXT} | sed -e 's/\.[^.]*$//')

for FILE_NAME in $FILE_LIST
do

CUDA_VISIBLE_DEVICES=1 python _SD_study.py \
    --input_image ${IMG_FOLDER}/${SEARCH_TEXT}/${FILE_NAME}.${IMG_FORMAT} \
    --results_folder output/${SEARCH_TEXT}_${FILE_NAME}/ \
    --manual_prompt \
    --prompt_str ${PROMPT} \

done