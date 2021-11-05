#!/usr/bin/bash

python predict.py --model 'resnet_rnn' --test_path '/Users/sjhuang/Documents/docs/dataset/test/rktc.padis.net.cn' \
--checkpoint_path 'checkpoints/crc_checkpoints/20211104/20211104_21213_resnet_rnn.pt'