#!/bin/bash

CUDA_VISIBLE_DEVICES=0,...,8 python3 src/main.py -sf -sf_num 50000 -cfg ./src/configs/artCIFAR10/ReACGAN-DiffAug.yaml -best -ckpt ./checkpoints/artCIFAR10-ReACGAN-DiffAug-train-2022_05_25_02_05_11 -data ./data -save ./out