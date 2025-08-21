#!/usr/bin/env bash
CSV_PATH=$1
NAME=${2:-vs_experiment}
GPUS=${3:-0}
CHECKPOINTS_DIR=${4:-./checkpoints}

CUDA_VISIBLE_DEVICES=${GPUS} python train_vs.py \
    --csv ${CSV_PATH} \
    --name ${NAME} \
    --gpu_ids ${GPUS} \
    --checkpoints_dir ${CHECKPOINTS_DIR} \
    --dataset_mode nifti \
    --model pix2pix3d_radiomics \
    --lambda_rad 10.0

CUDA_VISIBLE_DEVICES=0 python train_vs_end2end.py \
    --dataset_mode vs \
    --csv_file /path/to/vs.csv \
    --model pix2pix3d_radiomics \
    --name vs_radiomics \
    --dataroot . \
    --batch_size 1 \
    --load_size 64 \
    --crop_size 64 \
    --display_freq 50 \
    --lambda_rad 10.0 \
    --mask_aug dilate


