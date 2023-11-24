
IMG_FORMAT='png'

IMG_FOLDER='clip_retrieval/'
# SEARCH_TEXT='a_cat_and_a_dog'
# SEARCH_TEXT='a_cat_and_two_dogs'
# SEARCH_TEXT='a_dog_a_cat_and_a_child'
# SEARCH_TEXT='a_bird_a_bear_and_a_tree'
# SEARCH_TEXT='a_horse_and_a_sheep'
# SEARCH_TEXT='a_book_and_a_clock'
# SEARCH_TEXT='a_cat_and_a_bird'
# SEARCH_TEXT='a_cat_and_a_dog_on_the_grass'
# SEARCH_TEXT='a_cat_on_the_grass'
# SEARCH_TEXT='a_football_on_a_bench'
# SEARCH_TEXT='a_pizza_on_a_table'
# SEARCH_TEXT='a_man_playing_the_basketball'
# SEARCH_TEXT='a_cake_in_a_plate'
# SEARCH_TEXT='a_cat_and_a_ball'
# SEARCH_TEXT='a_keyboard_and_a_mouse'
# SEARCH_TEXT='a_piano_and_a_chair_in_the_room'
# SEARCH_TEXT='a_bird_and_a_dog'
# SEARCH_TEXT='a_dog_and_a_cat'
# SEARCH_TEXT='a_clock_and_a_book'
# SEARCH_TEXT='a_sheep_and_a_horse'
# SEARCH_TEXT='a_person_on_a_bike'
# SEARCH_TEXT='a_book_and_a_pen'
# SEARCH_TEXT='a_teapot_and_a_teacup'
# SEARCH_TEXT='a_rabbit_and_a_dog'
# SEARCH_TEXT='a_knife_and_a_fork'
# SEARCH_TEXT='a_cat_drinking_milk_from_a_bowl'
# SEARCH_TEXT='a_dog_chewing_on_a_bone'
# SEARCH_TEXT='a_kitten_and_a_rabbit'
# SEARCH_TEXT='a_kitten_and_a_bird'
# SEARCH_TEXT='a_puppy_and_a_bird'
# SEARCH_TEXT='a_puppy_and_a_rabbit'
# SEARCH_TEXT='a_boy_and_a_girl'
SEARCH_TEXT='a_cat'

# 0 6 9 21 42 50 51  54 55 58 61 63 64  
# 6 77 81
# 6 13 65 66 15 22 53
# 35
# 190 293 6 162 90 234 238 240 
# 21 177 168 73 130 64 76 88 97 106 108 1 48
# 234 173 250 154 167 15 41 44 151 201 8 171 19
# 69 56 45 44 76 83 82 46
# 2 77 76 
# 0 6 
# 27 93 96 
# 158 147 102 113 154 152 88 127 60 27
# 123 188 202 21 217 237 278
# 0 100 103 107 138 14 158 162 170 174 19 203 206 26 46 48 52 60 67 7 72 73 79 80
# 192 196 253 87
# 100 131 15 166
# 100 136 149 23 27 36 47 71 79
# 11 22 23 27 33 48 66 77
# 1 10 107 111 114 123 124 127 132 17 179 182 183 188 193 196 201 219 228 23 27 33 46 49 51 59 64 66 80 81 89 93 95 96
# 106 109 130 17 22 24 29 34 38 42 50 64 9
# 10 141 218 261 29 297 299 3 61 71 89
# 1 101 123 157 179 27 37 40
# 0 103 117 13 152 16 17 21 222 231 242 246 271 31 33 52 88
# 105 112 129 211
#  101 127 148 17 24 36 45 46 47 62 67 68 77
#  138 162 30 4 70 89
# 18 29 38
# 1 11 110 138 163 227 230 233 38 51 58 72 98
# 1 10 100 122 123 126 152 201 203 206 258 62 69 70 80 81 86 88 90
# 108 116 134 145 16 20 23 43 6 62 67 68 69 7 72 74  83 93 97

FILE_LIST=$(ls -1 clip_retrieval/${SEARCH_TEXT} | sed -e 's/\.[^.]*$//')


for FILE_NAME in $FILE_LIST
do

CUDA_VISIBLE_DEVICES=2 python _ddim_inversion.py \
    --input_image "${IMG_FOLDER}/${SEARCH_TEXT}/${FILE_NAME}.${IMG_FORMAT}" \
    --results_folder "output/${SEARCH_TEXT}_${FILE_NAME}/" \

done