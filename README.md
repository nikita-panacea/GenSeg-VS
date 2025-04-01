# Generative AI Enables Medical Image Segmentation in Ultra Low-Data Regimes

## Requirements

Python 3.9 and Pytorch 1.13.1 and CUDA 11.6 are recommended. A local conda env can be create using the following:

```bash
bash env.sh
conda activate semantic
```

## Datasets

We use [JSRT](http://db.jsrt.or.jp/eng.php) as the in-domain dataset to train and evaluate the model. Further, we use [NLM(MC)](https://drive.google.com/file/d/1cBKYYtlNIsOjxaeo9eQoCr9RL13CdAma/view?usp=share_link) and [NLM(SZ)](https://drive.google.com/drive/folders/1TewVvRjoZ1Ynm9AVsVzauGmlQYjA1QDH?usp=share_link), as the out-of-domain datasets for model evaluation. (*For some image examples, their lung segmentation masks are divided into the right and the left lung mask. For these, the masks need to be combined first.*)

Download the [data](https://drive.google.com/file/d/1L3Hj-G5g5g7WLcK85LJ6PkslEsJun2o2/view?usp=sharing) and place it in the [data](./data) folder. Dataset tree structure example:

```bash
data/JSRT
├── Images
│   ├── JPCLN001.png
│   ├── JPCLN002.png
│   ├── ...
├── Masks
│   ├── JPCLN001.gif
│   ├── JPCLN002.gif
│   ├── ...
project code
├── ...
```

## Training and Testing

We pre-train the GAN-based augmentation model on the train and val sets of the in-domain dataset followed by training both augmentation and semantic segmentation models end-to-end on the in-domain dataset. Finally, we test the trained models on the out-of-domain datasets. The results on the test set of both in-domain and out-of-domain datasets are shown using wandb during training. The training process needs about 1.5 hours on the NVIDIA A100 device with 40G memory.

To train the models from scratch, use the following command (Related configurations of model path should be changed mutually):

```
# Pre-train the augmentation model
bash scripts/train_pix2pix_jsrt.sh

# Train the segmentation based on our framework
bash scripts/train_end2end_jsrt.sh

# Inference the trained segmentation model
bash scripts/test_lung.sh

```

## Pre-trained model

Models pre-trained on the JSRT dataset (*trained with 9 labeled data examples*) are available through the following links: [Pix2Pix-generator](https://drive.google.com/file/d/1dkl55IFI_sAUCVQAPKq67aKvY_8p4yn3/view?usp=share_link) | [Pix2Pix-discriminator](https://drive.google.com/file/d/1cOAG_tf6bdVfqO424a6IIyYaEHXXji8n/view?usp=share_link) | [U-Net](https://drive.google.com/file/d/1V8mrJYAwE22Y3svy21bV2AjKvEMrsQ8G/view?usp=share_link)


## Extension for 3D medical image segmentaiton

We extended GenSeg for [3D medical image segmentation](./GenSeg-3D)


## Extension on diffusion-based and VAE-based mask-to-image generative models

We verify the effectiveness of GenSeg with diffusion-based and VAE-based mask-to-image generative models by replacing the original Pix2Pix model by [BBDM](./BBDM) and [Soft-intro VAE](./Soft-intro_VAE).

## Exploration on generating image-mask pairs simultaneously

We explore the effectiveness of synthesizing image-mask pairs simultaneously with WGAN-GP.

## Citation
If you find this project useful in your research, please consider citing:
```bash
@article{zhang2024generative,
  title={Generative AI Enables Medical Image Segmentation in Ultra Low-Data Regimes},
  author={Zhang, Li and Jindal, Basu and Alaa, Ahmed and Weinreb, Robert and Wilson, David and Segal, Eran and Zou, James and Xie, Pengtao},
  journal={medRxiv},
  pages={2024--08},
  year={2024},
  publisher={Cold Spring Harbor Laboratory Press}
}
```

## Code Dependencies

Our code is based on the following repositories: [Pix2Pix model](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/master/models) | [Betty framework](https://github.com/leopard-ai/betty)

## License

GenSeg is licensed under the [Apache 2.0 License](LICENSE).
