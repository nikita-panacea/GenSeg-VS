import os
import sys
sys.path.append('.')
import time
import logging

import wandb
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from options.train_options import TrainOptions
from models_pix2pix import create_model, networks
from unet import UNet
from unet.evaluate import evaluate
from deeplab import DeepLabV3

from util.Radpretation_loader import RadpretationDataset
from util.dice_score import dice_loss, multiclass_dice_coeff

from betty.engine import Engine
from betty.configs import Config, EngineConfig
from betty.problems import ImplicitProblem

# -------------------------------
# 1.   Parse options & constants
# -------------------------------
opt = TrainOptions().parse()

# Override a few defaults for Radpretation convenience
opt.input_nc = 1  # grayscale
opt.output_nc = 1
opt.classes = 6   # 0 background + 5 bleed types

assert opt.cuda_index == int(opt.gpu_ids[0]), 'GPU types should be the same'
device = torch.device(f'cuda:{opt.cuda_index}' if torch.cuda.is_available() else 'cpu')

save_path = os.path.join('./checkpoint_radpretation/',
                         time.strftime('%Y%m%d-%H%M%S-radpretation'))
os.makedirs(save_path, exist_ok=True)
unet_save_path = os.path.join(save_path, f'{opt.seg_model}.pth')

# -------------------------------
# 2.   Logging / W&B
# -------------------------------
logger = wandb.init(
    project='GenSeg-Radpretation',
    name=f'{opt.seg_model}-{opt.loss_lambda}',
    resume='allow',
    anonymous='must'
)
logger.config.update(vars(opt))
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# -------------------------------
# 3.   Create models
# -------------------------------
model = create_model(opt)
model.setup(opt)

if opt.seg_model == 'unet':
    seg_net = UNet(n_channels=opt.output_nc, n_classes=opt.classes, bilinear=opt.bilinear)
else:
    seg_net = DeepLabV3(num_classes=opt.classes)
seg_net = seg_net.to(device=device)

# -------------------------------
# 4.   Optimisers
# -------------------------------
optimizer_unet = optim.Adam(seg_net.parameters(), lr=opt.unet_learning_rate)
scheduler_unet = optim.lr_scheduler.ReduceLROnPlateau(optimizer_unet, 'max', patience=5)

# -------------------------------
# 5.   Dataloaders
# -------------------------------
# Radpretation expected directory structure:
#   dataroot/Images  *.png  (2D slices)
#   dataroot/Masks   per-volume subfolders, per-class subfolders, *.png (segmentation masks)

dataset = RadpretationDataset(os.path.join(opt.dataroot, 'Images'), os.path.join(opt.dataroot, 'Masks'), scale=1.0)

# Build custom few-shot split if requested
if opt.few_shot_per_class > 0:
    # Map class -> list of dataset indices containing that class
    cls_indices = {c: [] for c in range(1, opt.classes)}  # exclude background 0
    for idx in range(len(dataset)):
        mask = dataset[idx]['mask'][0]  # tensor C=1 H W
        present = torch.unique(mask).tolist()
        for cls in present:
            if cls != 0 and cls in cls_indices:
                cls_indices[cls].append(idx)
    train_ids = []
    rng = torch.Generator().manual_seed(42)
    for c, inds in cls_indices.items():
        if len(inds) < opt.few_shot_per_class:
            train_ids.extend(inds)
        else:
            perm = torch.randperm(len(inds), generator=rng)
            train_ids.extend([inds[i] for i in perm[:opt.few_shot_per_class]])
    train_set = torch.utils.data.Subset(dataset, train_ids)
    val_set = torch.utils.data.Subset(dataset, list(set(range(len(dataset))) - set(train_ids)))
else:
    n_val = int(len(dataset) * opt.val_percent / 100)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=opt.batch_size, shuffle=False, num_workers=4, pin_memory=True)

# -------------------------------
# 6.   Training Loop
# -------------------------------
# (Use the same Engine/Betty structure as in train_end2end_bhsd.py)

# -------------------------------
# 7.   Betty Problems definitions
# -------------------------------
criterion_seg = nn.CrossEntropyLoss()
criterion_gan = networks.GANLoss(opt.gan_mode).to(device)
criterion_l1 = nn.L1Loss()

class Generator(ImplicitProblem):
    def training_step(self, batch):
        real_mask = batch['mask_pix2pix'].to(device=device, dtype=torch.float32)
        real_image = batch['image_pix2pix'].to(device=device, dtype=torch.float32)
        fake_image = self.module(real_mask)
        pred_fake = self.netD(torch.cat((real_mask, fake_image), 1))
        loss_gan = criterion_gan(pred_fake, True)
        loss_l1 = criterion_l1(fake_image, real_image) * opt.loss_lambda
        return loss_gan + loss_l1

class Discriminator(ImplicitProblem):
    def training_step(self, batch):
        real_mask = batch['mask_pix2pix'].to(device=device, dtype=torch.float32)
        real_image = batch['image_pix2pix'].to(device=device, dtype=torch.float32)
        fake_image = self.netG(real_mask).detach()

        pred_fake = self.module(torch.cat((real_mask, fake_image), 1))
        loss_fake = criterion_gan(pred_fake, False)

        pred_real = self.module(torch.cat((real_mask, real_image), 1))
        loss_real = criterion_gan(pred_real, True)
        return 0.5 * (loss_fake + loss_real)

class UNetProblem(ImplicitProblem):
    def training_step(self, batch):
        images = batch['image'].to(device=device, dtype=torch.float32)
        masks_gt = batch['mask'].squeeze(1).to(device=device, dtype=torch.long)  # [B,H,W]
        masks_pred = self.module(images)
        loss = criterion_seg(masks_pred, masks_gt)
        loss_dice = dice_loss(F.softmax(masks_pred, dim=1), F.one_hot(masks_gt, num_classes=opt.classes).permute(0,3,1,2).float(), multiclass=True)
        return loss + loss_dice

class Arch(ImplicitProblem):
    def training_step(self, batch):
        images = batch['image'].to(device=device, dtype=torch.float32)
        masks_gt = batch['mask'].squeeze(1).to(device=device, dtype=torch.long)
        masks_pred = self.unet(images)
        loss = criterion_seg(masks_pred, masks_gt)
        return loss

class GenSegEngine(Engine):
    @torch.no_grad()
    def validation(self):
        val_score = evaluate(self.unet.module, val_loader, device, opt.amp)
        logger.log({'val_dice': val_score})
        scheduler_unet.step(val_score)
        # Save best model
        if not hasattr(self, '_best') or val_score > self._best:
            self._best = val_score
            torch.save(seg_net.state_dict(), unet_save_path)
            logging.info(f'New best model saved with Dice {val_score:.4f}')

# -------------------------------
# 8.   Instantiate Betty problems & engine
# -------------------------------
outer_cfg = Config(retain_graph=True)
inner_cfg = Config(type='darts', unroll_steps=opt.unroll_steps)
engine_cfg = EngineConfig(valid_step=opt.display_freq * opt.unroll_steps, train_iters=opt.n_epochs, roll_back=True)

netG_problem = Generator(
    module=model.netG,
    optimizer=model.optimizer_G,
    train_data_loader=train_loader,
    name='netG',
    config=inner_cfg,
    device=device,
)
netD_problem = Discriminator(
    module=model.netD,
    optimizer=model.optimizer_D,
    train_data_loader=train_loader,
    name='netD',
    config=inner_cfg,
    device=device,
)
unet_problem = UNetProblem(
    module=seg_net,
    optimizer=optimizer_unet,
    train_data_loader=train_loader,
    name='unet',
    config=inner_cfg,
    device=device,
)
arch_problem = Arch(
    module=seg_net,
    optimizer=optim.Adam(networks.arch_parameters(), lr=opt.arch_lr),
    train_data_loader=val_loader,
    name='arch',
    config=outer_cfg,
    device=device,
)

problems = [netG_problem, netD_problem, unet_problem, arch_problem]

l2u = {
    netG_problem: [unet_problem, arch_problem],
    netD_problem: [unet_problem]
}
u2l = {
    arch_problem: [netG_problem]
}
deps = {"l2u": l2u, "u2l": u2l}

engine = GenSegEngine(config=engine_cfg, problems=problems, dependencies=deps)
engine.run()

torch.save(seg_net.state_dict(), os.path.join(save_path, 'final_unet.pth'))
logger.finish() 