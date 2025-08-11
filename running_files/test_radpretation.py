import os
import sys
sys.path.append('.')
import argparse
import logging

import torch
from torch.utils.data import DataLoader

from options.train_options import TrainOptions
from unet import UNet
from deeplab import DeepLabV3
from unet.evaluate import evaluate

from util.Radpretation_loader import RadpretationDataset

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def main():
    opt = TrainOptions().parse()
    device = torch.device(f'cuda:{opt.gpu_ids[0]}' if torch.cuda.is_available() else 'cpu')

    # Define model
    if opt.seg_model == 'unet':
        net = UNet(n_channels=1, n_classes=opt.classes)
    else:
        net = DeepLabV3(num_classes=opt.classes)
    net = net.to(device)

    # Load checkpoint
    assert os.path.exists(opt.model_dir), f'Checkpoint {opt.model_dir} not found.'
    net.load_state_dict(torch.load(opt.model_dir, map_location=device))

    # Dataset
    dataset = RadpretationDataset(os.path.join(opt.dataroot, 'Images'), os.path.join(opt.dataroot, 'Masks'), scale=1.0)
    loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    score = evaluate(net, loader, device, opt.amp)
    print(f'Dice score on Radpretation test set: {score:.4f}')

if __name__ == '__main__':
    main() 