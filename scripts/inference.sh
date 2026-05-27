#!/bin/bash
# Standard inference (baseline, no speculative decoding)
python -u inference.py \
        --use_speculative_decoding \
        --compare_AR_SSD \
        --test_fname /data/wangzhichao/projects/SSD_adapters/data/annotations/2fps/ego_dataset.json \
        --output_fname ./outputs/text_4-3_0.6_stage2_1.jsonl \
        --device cuda:4 \
        --exit_layer 4 \
        --num_adapter_layers 3 \
        --adapter_path /data/wangzhichao/projects/SSD_adapters/adapter_checkpoints/multiturn_stage2_4-3_epoch019/epochs/epoch000_valtop10.6622_valdraft11.0000_vallongdraft11.0000_valoverlap0.5973_valloss1.7719 \
        --speculative_threshold 0.6 \
    > ./logs/text_4-3_0.6_stage2_1_all.log 2>&1