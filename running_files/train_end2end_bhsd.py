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

from util.BHSD_loader import BHSDDataset
from util.dice_score import dice_loss, multiclass_dice_coeff

from betty.engine import Engine
from betty.configs import Config, EngineConfig
from betty.problems import ImplicitProblem


# -------------------------------
# 1.   Parse options & constants
# -------------------------------
opt = TrainOptions().parse()

# Override a few defaults for BHSD convenience
opt.input_nc = 1  # CT slices are single–channel
opt.output_nc = 1
opt.classes = 6   # 0 background + 5 bleed types

assert opt.cuda_index == int(opt.gpu_ids[0]), 'GPU types should be the same'
device = torch.device(f'cuda:{opt.cuda_index}' if torch.cuda.is_available() else 'cpu')

save_path = os.path.join('./checkpoint_bhsd/',
                         f"fewshot_{opt.few_shot_per_class}" if opt.few_shot_per_class>0 else '',
                         time.strftime('%Y%m%d-%H%M%S-bhsd'))
os.makedirs(save_path, exist_ok=True)
unet_save_path = os.path.join(save_path, f'{opt.seg_model}.pth')

# -------------------------------
# 2.   Logging / W&B
# -------------------------------
logger = wandb.init(
    project='GenSeg-BHSD',
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
# BHSD expected directory structure:
#   dataroot/Images  *.nii.gz  (CT volumes)
#   dataroot/Masks   *.nii.gz  (segmentation volumes matching file names)

dataset = BHSDDataset(os.path.join(opt.dataroot, 'Images'), os.path.join(opt.dataroot, 'Masks'), scale=1.0)

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
            logging.warning(f"Class {c} has only {len(inds)} slices, requested {opt.few_shot_per_class}")
        selected = torch.randperm(len(inds), generator=rng)[:opt.few_shot_per_class].tolist()
        train_ids.extend([inds[i] for i in selected])
    # ensure unique
    train_ids = list(set(train_ids))
    remaining = [i for i in range(len(dataset)) if i not in train_ids]
    # simple split of remaining into val/test
    n_val = int(len(remaining)* opt.val_percent/100)
    n_test = int(len(remaining)* opt.test_percent/100)
    val_ids = remaining[:n_val]
    test_ids= remaining[n_val:n_val+n_test]
    train_set = torch.utils.data.Subset(dataset, train_ids)
    val_set   = torch.utils.data.Subset(dataset, val_ids)
    test_set  = torch.utils.data.Subset(dataset, test_ids)
    n_train = len(train_set)
else:
    n_val = int(len(dataset) * opt.val_percent / 100)
    n_test = int(len(dataset) * opt.test_percent / 100)
    n_train = len(dataset) - n_val - n_test
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42))


loader_args = dict(batch_size=opt.batch_size, num_workers=4, pin_memory=True)
train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_args)
val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)
test_loader = DataLoader(test_set, shuffle=False, drop_last=False, **loader_args)

logging.info(f'Training slices: {n_train}, Validation slices: {n_val}, Test slices: {n_test}')

# -------------------------------
# 6.   Loss functions
# -------------------------------
criterion_seg = nn.CrossEntropyLoss()
criterion_gan = networks.GANLoss(opt.gan_mode).to(device)
criterion_l1 = nn.L1Loss()

# -------------------------------
# 7.   Betty Problems definitions
# -------------------------------
class Generator(ImplicitProblem):
    def training_step(self, batch):
        real_mask = batch['mask_pix2pix'].to(device=device, dtype=torch.float32)
        real_image = batch['image_pix2pix'].to(device=device, dtype=torch.float32)
        fake_image = self.module(real_mask)
        pred_fake = self.netD(torch.cat((real_mask, fake_image), 1))
        loss_gan = criterion_gan(pred_fake, True)
        loss_l1 = criterion_l1(fake_image, real_image) * opt.lambda_L1
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
        masks_pred = seg_net(images)

        loss = criterion_seg(masks_pred, masks_gt)
        loss_dice = dice_loss(F.softmax(masks_pred, dim=1), F.one_hot(masks_gt, num_classes=opt.classes).permute(0,3,1,2).float(), multiclass=True)
        return loss + loss_dice

class Arch(ImplicitProblem):
    def training_step(self, batch):
        # Validation loss for architecture search (optional)
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

# Low-to-Upper: how lower-level problems influence upper ones
l2u = {
    netG_problem: [unet_problem, arch_problem],
    netD_problem: [unet_problem]
}
# Upper-to-Low: which lower problems an upper-level one controls
u2l = {
    arch_problem: [netG_problem]
}

deps = {"l2u": l2u, "u2l": u2l}

engine = GenSegEngine(config=engine_cfg, problems=problems, dependencies=deps)
engine.run()

torch.save(seg_net.state_dict(), os.path.join(save_path, 'final_unet.pth'))
logger.finish() 