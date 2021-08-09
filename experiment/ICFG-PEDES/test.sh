#!/bin/bash

cd src

python test.py --model_name 'SSAN' \
--GPU_id 0 \
--part 6 \
--lr 0.001 \
--dataset 'ICFG-PEDES' \
--dataroot '../dataset/ICFG-PEDES/' \
--vocab_size 2500 \
--feature_length 1024 \
--mode 'test'
