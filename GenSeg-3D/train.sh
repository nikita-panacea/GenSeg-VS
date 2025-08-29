CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataroot /path/to/dataset/root \
    --dataset_mode vscsv \
    --csv_file /path/to/your/vs_dataset.csv \
    --model pix2pix3d \
    --name vs_pix2pix3d_experiment \
    --batch_size 1 \
    --gpu_ids 0 \
    --phase train \