#!/usr/bin/bash

python predict.py --model 'resnet_rnn' --test_path '/Users/sjhuang/Documents/docs/dataset/test' \
--checkpoint_path 'checkpoints/crc_checkpoints/20211108/20211108_25898_resnet_rnn.pt' \
--multi True