#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_PATH=/data/wangzhichao/projects/MMDuet2/ckpt/MMDuet2_ckpt
DATA_DIR=/data/wangzhichao/projects/SSD_test/AAAA/datasets/81012_layers_no_reply_0.01
ADAPTER_PATH=/data/wangzhichao/projects/SSD_test/AAAA/adapter_checkpoints/offline_12_mrope_kl+ce_re/epochs/epoch060_top10.8985_draft11.0000_longdraft11.0000_overlap0.8407_loss0.3430
EXIT_LAYER=12

CUDA_VISIBLE_DEVICES=6 python evaluate_adapter_ckpt.py \
    --basepath $MODEL_PATH \
    --datadir $DATA_DIR \
    --adapter_path $ADAPTER_PATH \
    --exit_layer $EXIT_LAYER \
    --device cuda:0 \
    --dtype bf16 \
    --kl_temperature 1.0 \
    --prefix_tokens 4 \
    --prefix_start 1 \
    --min_reply_tokens 6 \
    --output_jsonl ./outputs/adapter_train_eval.jsonl

