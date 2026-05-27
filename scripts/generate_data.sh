CUDA_VISIBLE_DEVICES=6 python generate_training_data.py \
    --model_path /data/wangzhichao/projects/MMDuet2/ckpt/MMDuet2_ckpt \
    --data_path /data/wangzhichao/projects/SSD_adapters/data/annotations/adapter/all.jsonl \
    --output_dir /data/wangzhichao/projects/SSD_adapters/datasets/multiturn_4_noreplymask0.1 \
    --exit_layers 4 \
    --no_reply_mask_weight 0.1 \
    --include_im_end_in_loss \
    --min_real_reply_tokens 20 \
    --max_seq_len  4096 \
    --gpu cuda:0