
al_COSINE=0.6

CUDA_VISIBLE_DEVICES=8 \
python seg_null_textinv.py \
        --results_folder output/catdog/ \
        --alpha_cos  ${al_COSINE} \
