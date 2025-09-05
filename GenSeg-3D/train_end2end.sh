CUDA_VISIBLE_DEVICES=0 python test_vs_end2end.py \
    --csv '/home/ubuntu/Documents/Nafisha/GenSeg-VS/GenSeg_VS.csv' \
    --dataroot /home/ubuntu/Documents/Nafisha/VS_data_nifti_Genseg/ \
    --dataset_mode vscsv \
    --model pix2pix3d \
    --name end2end-vs-128 \
    --crop_size 128 \
    --load_size 128 \
    --display_winsize 128 \