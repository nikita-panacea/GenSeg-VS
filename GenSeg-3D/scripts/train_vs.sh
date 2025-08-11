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


