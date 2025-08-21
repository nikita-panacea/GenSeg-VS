CUDA_VISIBLE_DEVICES=1 python GenSeg-3D/train_vs_radiomics.py \
    --csv /data/li/Pix2PixNIfTI/data_liver/vs_dataset.csv \ 
    --name vs_exp \ 
    --checkpoints_dir ./pix2pix_vs_checkpoints \ 
    --gpu_ids 0 \ 
    --batch_size 1 \ 
    --n_epochs 500 \ 
    --lambda_pyrad 100.0 \ 
    --features original_shape_Maximum2DDiameterRow,original_shape_SurfaceArea,original_glszm_LargeAreaHighGrayLevelEmphasis,original_shape_SurfaceVolumeRatio,original_shape_MajorAxisLength,original_shape_Maximum3DDiameter