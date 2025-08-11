#!/bin/bash
# Evaluate a trained GenSeg segmentation model on the BHSD test set
# Specify --model_dir to the checkpoint produced by train_end2end_bhsd.sh

python running_files/test_bhsd.py \
    --model pix2pix \
    --is_train False \
    --cuda True \
    --gpu_ids 0 \
    --cuda_index 0 \
    --dataroot /path/to/BHSD \
    --amp \
    --batch_size 4 \
    --seg_model unet \
    --classes 6 \
    --model_dir /path/to/checkpoint/unet.pth 