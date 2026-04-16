
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.6"
export CUDA_VISIBLE_DEVICES=0

python train_end2end.py \
    --csv '/home/ubuntu/Documents/Nafisha/GenSeg-VS/GenSeg_VS_with_brainmask.csv' \
    --dataroot /home/ubuntu/Documents/Nafisha/VS_data_nifti_Genseg/ \
    --dataset_mode vscsv \
    --model pix2pix3d \
    --name end2end-vs-128-model \
    --crop_size 128 \
    --load_size 128 \
    --display_winsize 128 \
    --ngf 64 \
    --ndf 64 \
    "$@"

 # --fp16 True \