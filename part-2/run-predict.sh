#!/usr/bin/env sh

python predict.py /home/workspace/aipnd-project/flowers/valid/100/image_07895.jpg /home/workspace/paind-project/checkpoints/vgg13-accuracy-85.53.pth --gpu --category_names /home/workspace/aipnd-project/cat_to_name.json --top_k 3