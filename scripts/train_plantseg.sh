#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs
now=$(date +"%Y%m%d_%H%M%S")

CUDA_VISIBLE_DEVICES=0 python train_stage2.py \
    --dataset plantseg \
    --plantseg_root ../plantseg \
    --caption_index 3 \
    --batch_size 8 \
    --size 320 \
    --test_split val \
    --bert_tokenizer clip \
    --backbone clip-RN50 \
    --max_query_len 77 \
    --epoch 50 \
    --clip_model_path ./weights/pretrained/RN50.pt \
    --pretrained_checkpoint ./weights/pretrained/stage2_refcocog_umd.pth \
    --output ./weights/plantseg/stage2 \
    --board_folder ./output/plantseg/tensorboard \
    2>&1 | tee logs/train_${now}_plantseg_stage2.txt
