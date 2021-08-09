#!/bin/bash

cd src

python train.py --model_name 'SSAN' \
--GPU_id 0 \
--part 6 \
--lr 0.001 \
--dataset 'CUHK-PEDES' \
--epoch 60 \
--dataroot '../dataset/CUHK-PEDES/' \
--class_num 11000 \
--vocab_size 5000 \
--feature_length 1024 \
--mode 'train' \
--batch_size 64 \
--cr_beta 0.1
