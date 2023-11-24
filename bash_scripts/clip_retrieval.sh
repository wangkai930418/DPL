
for PROMPT in   "a cat in a background" \
                # "a horse" \
                # "a zebra" \
                # "a tiger" \
                # "a lion" \
                # "a monkey" \
                # "a bacon and a pancake" \
                # "a burger and fries" \
                # "a steak and a potato" \
                # "a beer and a pizza" \
                # "a monkey eating a banana" \
                # "a cow chewing on hay in a barn" \
                # "a lion tearing into the flesh of its prey" \
                
do

python _clip_retrieval.py \
    --prompt_str "${PROMPT}" \
    --results_folder ./filter_clip_retrieval \

done