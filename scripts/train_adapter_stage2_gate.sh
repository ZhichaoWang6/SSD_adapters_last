#!/bin/bash
# Stage 2 (GATE): train the adapter as a proactive-response trigger on the
# ego streaming GT data (egoexo4d + egoexolearn, --keep_source_assistant).
#
# Differs from scripts/train_adapter_stage2.sh (which targets lossless
# drafting) in three gate-specific ways:
#   --ce_target gt        learn the REAL NO REPLY / content labels, not teacher
#   --prefix_ce_start 0   supervise EACH turn's first token (the trigger);
#                         the old stage2 used 1, which deliberately SKIPS it
#   --kl_weight 0.0       no teacher KL; pure GT cross-entropy
# Also: the .ckpt data must be generated with --no_reply_mask_weight 1.0
#       (NO REPLY is the target here, not noise).
#
# Fill in the four paths below. RESUME_ADAPTER should point at your stage-1
# (pure-text, teacher) adapter checkpoint's adapter_model.bin. Leave it empty
# for a one-stage run (worse content drafting; see notes in chat).

BASE_MODEL=/data/wangzhichao/projects/MMDuet2/ckpt/MMDuet2_ckpt
DATA_DIR=/path/to/ego_gate_ckpts                 # generated with --no_reply_mask_weight 1.0
OUT_DIR=/path/to/adapter_checkpoints/stage2_gate
RESUME_ADAPTER=/path/to/stage1_text/epochs/epochXXX/adapter_model.bin

# exit_layer / num_adapter_layers MUST match the stage-1 checkpoint and the
# data you generated (generate_training_data_multiturn.py --exit_layers).
EXIT_LAYER=2
NUM_ADAPTER_LAYERS=1

CUDA_VISIBLE_DEVICES=6 accelerate launch \
    --num_processes 1 --num_machines 1 --mixed_precision bf16 \
    train_adapter.py \
    --basepath "${BASE_MODEL}" \
    --datadir "${DATA_DIR}" \
    --outdir "${OUT_DIR}" \
    --resume_adapter "${RESUME_ADAPTER}" \
    --exit_layer ${EXIT_LAYER} \
    --num_adapter_layers ${NUM_ADAPTER_LAYERS} \
    --ce_target gt \
    --prefix_ce_start 0 \
    --prefix_ce_tokens 9999 \
    --prefix_ce_decay 1.0 \
    --prefix_ce_weight 1.0 \
    --kl_weight 0.0 \
    --lr 5e-6 \
    --bs 1 \
    --gradient_accumulation_steps 32 \
    --num_epochs 20 \
    --num_warmup_steps 30 \
    --grad_clip 1.0 \
    --save_freq 1 \
    --min_mask_tokens 0 \
    --val_ratio 0.05 \
    --val_seed 42
