
IMG_FORMAT='jpg'

IMG_FOLDER='coco_seginv/'

OBJECT='cat_dog'
OBJECT='apple_banana'
OBJECT='donkey_zebra'
OBJECT='person_dog_boat'


OBJECT='bench_cat'
OBJECT='boat'
OBJECT='cat'
OBJECT='elephant'
OBJECT='zebra'
OBJECT='dog_frisbee'
OBJECT='airplane'
OBJECT='bear'

# SEARCH_TEXT='a_cat'
# SEARCH_TEXT='a_cat_in_a_background'
SEARCH_TEXT='val2014/'
# SEARCH_TEXT='train2014/'

# PROMPT='cat'

# for OBJECT in 'bench_cat' 'boat'  'cat'   'elephant'  'zebra'
for OBJECT in 'giraffe' 

do

FILE_LIST=$(ls -1 ${IMG_FOLDER}/${OBJECT}/${SEARCH_TEXT} | sed -e 's/\.[^.]*$//')
echo ${FILE_LIST}

for FILE_NAME in $FILE_LIST
do

echo ${FILE_NAME}

CUDA_VISIBLE_DEVICES=2 python _2_SD_study.py \
    --input_image ${IMG_FOLDER}/${OBJECT}/${SEARCH_TEXT}/${FILE_NAME}/${FILE_NAME}.${IMG_FORMAT} \
    --results_folder ${IMG_FOLDER}/${OBJECT}/${SEARCH_TEXT}/ \
    # --manual_prompt \
    # --prompt_str ${PROMPT} \

done
done