
FILE_NAME="catdog"

CUDA_VISIBLE_DEVICES=1 python null_attend_textinv.py \
    --input_image "images/${FILE_NAME}.jpg" \
    --prompt_str "a cat and a dog" \
    --results_folder "output/${FILE_NAME}/" \
    --negative_guidance_scale 7.5 \
    --null_inner_steps 21 \
    --attn_inner_steps 11 \
    --indices_to_alter 2 5 \
    --max_iter_to_alter 55 \
    --lam_maxattn 1.0 \
    --lam_entropy 0.0 \
    --lam_cosine 0.1 \
    --lr_cond 0.01 \
    --print_freq 10 \
    # | tee -a output/${FILE_NAME}/log.txt