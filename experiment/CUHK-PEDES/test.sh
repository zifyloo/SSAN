#!/bin/bash

cd src

python test.py --model_name 'SSAN' \
--GPU_id 0 \
--part 6 \
--lr 0.001 \
--dataset 'CUHK-PEDES' \
--dataroot '../dataset/CUHK-PEDES/' \
--vocab_size 5000 \
--feature_length 1024 \
--mode 'test'
