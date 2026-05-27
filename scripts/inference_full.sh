python -u inference.py \
    --use_speculative_decoding \
    --compare_AR_SSD \
    --test_fname /data/wangzhichao/projects/SSD_test/AAAA/data/annotations/2fps/ego_dataset.json \
    --output_fname ./outputs/sanity_full_exit12.jsonl \
    --device cuda:6 \
    --exit_layer 12 \
    --adapter_path /tmp/no_adapter \
    --speculative_threshold 0.0 \
    --speculative_steps 6 \
  > ./logs/sanity_full_exit12.log 2>&1