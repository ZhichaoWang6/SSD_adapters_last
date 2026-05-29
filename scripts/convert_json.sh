#!/bin/bash
# Convert MMDuet2 SFT jsonl (messages/images + <image> placeholders) into the
# {question_id, conversation:[...]} format that generate_training_data_multiturn.py
# expects, KEEPING the original assistant replies (= ground truth).
#
# This is a pure CPU format conversion: NO model weights are loaded
# (the "Hello! loading Qwen2.5-VL model" lines are just import banners).
#
# Edit the paths below, then: bash scripts/convert_json.sh

set -euo pipefail

INPUT_JSONL="/data/wangzhichao/datasets/MMDuet2-data/sft/egoexo4d-half_multi_half_single_question-2_sec_per_frame-sft.jsonl"
OUTPUT_JSON="/data/wangzhichao/projects/SSD_adapters_last/data/annotations/ego_gt_conversation.json"
IMAGE_ROOT="/data/wangzhichao/datasets"      # frames live under here; adjust to your disk
STRIP_PREFIX="./data/datasets"               # stripped from the jsonl paths before joining IMAGE_ROOT

python -u generate_streaming_ar_from_sft.py \
    --keep_source_assistant \
    --input_jsonl "${INPUT_JSONL}" \
    --output_json "${OUTPUT_JSON}" \
    --image_root "${IMAGE_ROOT}" \
    --strip_prefix "${STRIP_PREFIX}"
    # Add --strip_ego_time_suffix ONLY if your frame folders are named
    # <video_id> WITHOUT the "-0s_180s" time suffix. If the folders on disk
    # literally contain "-0s_180s", do NOT add it.
    #
    # Add --renumber_frames_sequential ONLY if the subsampled frames on disk
    # are stored consecutively (000001, 000002, 000003, ...) rather than with
    # the source's original video frame indices (000001, 000003, 000005, ...).
    # It rewrites each filename to its 1-based position in the image list.
    # ls the frame folder first to check which layout you have.
