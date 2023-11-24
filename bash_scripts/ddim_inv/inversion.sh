
IMG_FORMAT='jpg'

FILE_NAME="cat_ball"
# PROMPT="a cat and a dog"
PROMPT="a small cat playing with a pink ball"

CUDA_VISIBLE_DEVICES=3 python _ddim_inversion.py \
    --input_image "images/${FILE_NAME}.${IMG_FORMAT}" \
    --prompt_str "${PROMPT}" \
    --results_folder "output/${FILE_NAME}/" \


# IMG_FORMAT='jpg'

# FILE_NAME="lexica_dog_cat"
# # PROMPT="a cat and a dog"
# PROMPT="a dog and a cat sitting on a gray background"

# CUDA_VISIBLE_DEVICES=3 python _ddim_inversion.py \
#     --input_image "images/${FILE_NAME}.${IMG_FORMAT}" \
#     --prompt_str "${PROMPT}" \
#     --results_folder "output/${FILE_NAME}/" \


# FILE_NAME="lexica_kitten_puppy" 
# PROMPT="a kitten and a puppy"

# CUDA_VISIBLE_DEVICES=3 python _ddim_inversion.py \
#     --input_image "images/${FILE_NAME}.${IMG_FORMAT}" \
#     --prompt_str "${PROMPT}" \
#     --results_folder "output/${FILE_NAME}/" \

# FILE_NAME="gen_1_a_cat_and_a_dog"
# PROMPT="a cat and a dog"

# CUDA_VISIBLE_DEVICES=3 python _ddim_inversion.py \
#     --input_image "images/${FILE_NAME}.png" \
#     --prompt_str "${PROMPT}" \
#     --results_folder "output/${FILE_NAME}/" \
    
# FILE_NAME="gen_3_a_cat_and_a_dog"
# PROMPT="a cat and a dog"

# CUDA_VISIBLE_DEVICES=3 python _ddim_inversion.py \
#     --input_image "images/${FILE_NAME}.png" \
#     --prompt_str "${PROMPT}" \
#     --results_folder "output/${FILE_NAME}/" \

# FILE_NAME="pear_apple"
# PROMPT="a pear and a apple"

#"coco_catdog1" "coco_catdog2" "coco_catdog3" "coco_catdog4"  
# "apple_potato_limon" "book_banana_keyboard" "book_cup_telephone" 


# for FILE_NAME in "apple_orange_pepper" 
# do
# PROMPT="a apple a orange and a pepper"
# CUDA_VISIBLE_DEVICES=7 python _ddim_inversion.py \
#     --input_image "images/${FILE_NAME}.jpg" \
#     --prompt_str "${PROMPT}" \
#     --results_folder "output/${FILE_NAME}/" \

# done

# for FILE_NAME in "apple_potato_limon" 
# do
# PROMPT="a apple a potato and a limon"
# CUDA_VISIBLE_DEVICES=7 python _ddim_inversion.py \
#     --input_image "images/${FILE_NAME}.jpg" \
#     --prompt_str "${PROMPT}" \
#     --results_folder "output/${FILE_NAME}/" \

# done

# for FILE_NAME in "book_banana_keyboard" 
# do
# PROMPT="a book a banana and a keyboard"
# CUDA_VISIBLE_DEVICES=7 python _ddim_inversion.py \
#     --input_image "images/${FILE_NAME}.jpg" \
#     --prompt_str "${PROMPT}" \
#     --results_folder "output/${FILE_NAME}/" \

# done

# for FILE_NAME in "book_cup_telephone" 
# do
# PROMPT="a book a cup and a telephone"
# CUDA_VISIBLE_DEVICES=7 python _ddim_inversion.py \
#     --input_image "images/${FILE_NAME}.jpg" \
#     --prompt_str "${PROMPT}" \
#     --results_folder "output/${FILE_NAME}/" \

# done