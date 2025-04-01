from utils.config import parse_args
from utils.data_loader import get_data_loader
# from models.gan import GAN
# from models.dcgan import DCGAN_MODEL
# from models.wgan_clipping import WGAN_CP
from models.wgan_gradient_penalty import WGAN_GP

from deeplab import *
from unet import UNet
from util import util
# from util.JSRT_loader import CarvanaDataset as BasicDataset
from util.ISIC_loader import CarvanaDataset as BasicDataset
from util.dice_score import dice_loss
from unet.evaluate import evaluate
import wandb
import sys
import os
import time
import logging
import torch
import torchvision
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image
from PIL import Image


def main(args):
    device = torch.device('cuda:1')
    
    model = WGAN_GP(args)
    model.G.load_state_dict(torch.load(os.path.join(os.getcwd(), 'generator_derm40.pkl')))    
    
    for i in range(100):
        z = torch.randn(1, 100, 1 ,1).to(device=device)
        augmented_data = model.G(z).squeeze()
        fake_image = augmented_data[:3,:,:].mul(255).to('cpu', torch.uint8).numpy().transpose(1,2,0)
        Image.fromarray(fake_image).save(f'./generated_images/{i}.png')
    
    print("finish")
if __name__ == '__main__':
    args = parse_args()
    print(args.cuda)
    main(args)
