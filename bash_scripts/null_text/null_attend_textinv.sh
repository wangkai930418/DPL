
FILE_NAME="catdog"
# PROMPT="a <cat-toy> and a <dog-toy>"

ATTN=1.0
ENTROPY=0.0
COSINE=0.02
THRESHOLD=0.3

python print_args.py \
    --input_image "images/${FILE_NAME}.jpg" \
    --prompt_str "a <cat-toy> and a <dog-toy>" \
    --results_folder "output/${FILE_NAME}/" \
    --negative_guidance_scale 7.5 \
    --null_inner_steps 51 \
    --attn_inner_steps 21 \
    --indices_to_alter 2 5 \
    --lam_maxattn ${ATTN} \
    --lam_entropy ${ENTROPY} \
    --lam_cosine ${COSINE} \
    --threshold_init ${THRESHOLD} \
    --print_freq 10 \
    --max_iter_to_alter 55 \
    --placeholder_token '<cat-toy>' '<dog-toy>' \
    --initializer_token 'cat' 'dog' \