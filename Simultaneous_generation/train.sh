CUDA_VISIBLE_DEVICE=1 python main.py \
    --model WGAN-GP  \
    --is_train True \
    --dataroot "/data2/li/workspace/data/ISIC2018" \
    --dataset JSRT \
    --generator_iters 40000 \
    --cuda True \
    --batch_size 8 \
    --image_size 64 \
    --channels 4