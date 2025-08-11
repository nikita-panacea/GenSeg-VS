#!/bin/bash
# End-to-end GenSeg training on the Brain Hemorrhage Segmentation Dataset (BHSD)
# Adjust --dataroot to point at the directory that contains the sub-folders "Images" and "Masks".
# You can modify other hyper-parameters as needed (epochs, batch-size, learning-rates …).

python running_files/train_end2end_bhsd.py \
    --model pix2pix \
    --is_train True \
    --cuda True \
    --gpu_ids 0 \
    --cuda_index 0 \
    --dataroot /path/to/BHSD \
    --amp \
    --loss_lambda 1.0 \
    --n_epochs 5000 \
    --lr 2e-4 \
    --arch_lr 1e-4 \
    --display_freq 10 \
    --classes 6 \
    --output_nc 1 \
    --input_nc 1 \
    --batch_size 4 \
    --seg_model unet \
    --unet_learning_rate 1e-4 \
"$@" 