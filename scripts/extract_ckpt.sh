#!/bin/bash
# Extract adapter training ckpts: run the base model forward over the GT
# conversation json and save per-conversation hidden states (the adapter's
# INPUT) + input_ids (GT labels) into one .ckpt per conversation.
#
# Inputs:
#   DATA_PATH  = GT conversation .json from generate_streaming_ar_from_sft.py
#                (run with --keep_source_assistant)
#   OUTPUT_DIR = where the data_{i}.ckpt files go
#
# Needs: a GPU and the actual frame images readable at the paths in the json.

BASE_MODEL=/data/wangzhichao/projects/MMDuet2/ckpt/MMDuet2_ckpt
DATA_PATH=/path/to/ego_gt_conversation.json
OUTPUT_DIR=/path/to/ego_gate_ckpts

python -u generate_training_data_multiturn.py \
    --model_path "${BASE_MODEL}" \
    --data_path "${DATA_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --exit_layers 2 \
    --no_reply_mask_weight 1.0 \
    --gpu cuda:6

# Multi-GPU sharding example (run each on a different GPU, same OUTPUT_DIR):
#   ... --start 0    --end 1100 --gpu cuda:0 &
#   ... --start 1100 --end 2145 --gpu cuda:1 &
