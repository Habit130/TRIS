#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs
now=$(date +"%Y%m%d_%H%M%S")
checkpoint_name="${1:-best_model.pth}"

CUDA_VISIBLE_DEVICES=0 python validate_plantseg.py \
    --dataset plantseg \
    --plantseg_root ../plantseg \
    --caption_index 3 \
    --batch_size 1 \
    --size 320 \
    --test_split test \
    --bert_tokenizer clip \
    --backbone clip-RN50 \
    --max_query_len 77 \
    --output ./weights/plantseg/stage2 \
    --resume \
    --pretrain "${checkpoint_name}" \
    --save_pred_masks \
    --mask_output_dir ./output/plantseg/test_masks/ann \
    --metrics_output ./output/plantseg/eval/test_metrics.json \
    2>&1 | tee logs/eval_${now}_plantseg.txt
