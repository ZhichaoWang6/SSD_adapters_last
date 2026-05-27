
# if you use multinode, please make sure the following parameters are set correctly
# export NNODES="${JOB_ARGS[0]}"
# export GPUS_PER_NODE="${JOB_ARGS[1]}"
# export NPROC_PER_NODE="${JOB_ARGS[1]}"
# export MASTER_ADDR="${JOB_ARGS[2]}"
# export MASTER_PORT="${JOB_ARGS[3]}"
# export NODE_RANK="${JOB_ARGS[4]}"


# make sure you load qwen-vl-utils from `MMDuet2/train/qwen_vl_utils` instead of pip, so we can use env vars to set num tokens and num frames
# 128 tokens per frame
export MAX_PIXELS=$((128*28*28))
export VIDEO_MAX_PIXELS=$MAX_PIXELS
export FPS_MAX_FRAMES=64        # For offline videos, real-time interaction is not important, so if the video is too long, the FPS can be appropriately reduced. Setting a maximum of 64 frames is sufficient.

export PYTHONPATH=$(pwd):$PYTHONPATH        # load some modules from this folder

which conda
which python

timestamp=$(date +"%Y%m%d-%H%M%S")

# get the latest checkpoint
# ----- set output dir here -----
output_dir=./ckpt/qwen2_5
mkdir -p $output_dir

swift sft \
    --model_type qwen2_5_vl \
    --save_only_model false \
    --save_strategy steps \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset \
    ./train/data/annotations/egoexo4d-half_multi_half_single_question-2_sec_per_frame-sft.jsonl \
    ./train/data/annotations/egoexolearn-half_multi_half_single_question-2_sec_per_frame-sft.jsonl \
    ./train/data/annotations/live_whisperx-half_multi_half_single_question-2_sec_per_frame-max_180s-sft-h5_images.jsonl \
    ./train/data/annotations/tarsier2_25k.jsonl \
    ./train/data/annotations/llava_video_25k.jsonl \
    --enable_cache true \
    --freeze_vit true \
    --freeze_aligner false \
    --logging_steps 20 \
    --learning_rate 1e-5 \
    --output_dir $output_dir \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --dataloader_num_workers 1 \
    --dataset_num_proc 4 \
    --num_train_epochs 1 \
    --train_type full \
    --eval_steps 400 \
    --save_steps 200 \
    --torch_dtype bfloat16 \
    --max_length 10000 \
    --warmup_ratio 0.05 \
    --truncation_strategy right \
    --deepspeed ./train/scripts/zero3.json \
    --attn_impl flash_attn \
    --model_kwargs '{"fps": 1}' \
    $extra_params \
    > ./nohup/$timestamp-node_$NODE_RANK.log 2>&1
