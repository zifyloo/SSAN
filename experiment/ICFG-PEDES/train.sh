#!/bin/bash

cd src

python train.py --model_name 'SSAN' \
--GPU_id 0 \
--part 6 \
--lr 0.001 \
--dataset 'ICFG-PEDES' \
--epoch 60 \
--dataroot '../dataset/ICFG-PEDES/' \
--class_num 3102 \
--vocab_size 2500 \
--feature_length 1024 \
--batch_size 64 \
--mode 'train' \
--cr_beta 0.1
