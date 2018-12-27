#!/usr/bin/env sh

python train.py /home/workspace/aipnd-project/flowers --arch vgg13 --epochs 3 --learning_rate 0.0000075 --gpu --save_dir checkpoints