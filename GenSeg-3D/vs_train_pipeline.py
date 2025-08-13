import os
import sys
sys.path.append('.')
import time
import shutil
import wandb
import logging
import imgaug as ia
import numpy as np
from PIL import Image
from imgaug import augmenters as iaa
from copy import deepcopy
 
import torch
import torchvision
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset
 
from util import util
from UNet3D.unet3d import UNet3D
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from models.networks import arch_parameters
from transforms import fake_transform
from util.util import zero_division
 
from betty.engine import Engine
from betty.configs import Config, EngineConfig
from betty.problems import ImplicitProblem
 
from denseNet import DenseNet3D
 
def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    loss = 0
 
    # iterate over the validation set
    with torch.no_grad():
        # for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        for i, batch in enumerate(dataloader):
            image, label = batch['B'], batch['label']
            if label.dim() == 1:
                label = label.unsqueeze(1)
            label_tensor = torch.tensor([label], dtype=torch.float32)
 
            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32)
 
            age_pred = net(image)
            # compute the Dice score
            loss += criterion(age_pred, label_tensor)
 
    net.train()
    # return JC_index / max(num_val_batches, 1)
    return loss.item()/ max(num_val_batches, 1)
 
opt = TrainOptions().parse()   # get training options
# config = get_config(opt)
device = torch.device('cuda:0')
save_path = './checkpoint_e2e/'+'end2end-liver-98-'+time.strftime("%Y%m%d-%H%M%S")
if not os.path.exists(save_path):
    os.mkdir(save_path)
densenet_save_path = save_path+'/densenet.pkl'  
 
##### Initialize logging #####
# logger = wandb.init(project='end2end-unet-ISIC', name="unet-200", resume='allow', anonymous='must')
logger = wandb.init(project='end2end-hippo', name="e2e-liver-98",
                    resume='allow', anonymous='must', mode='disabled')
logger.config.update(vars(opt))
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
 
##### create models: pix2pix, unet #####
model = create_model(opt)      # create a model given opt.model and other options
model.setup(opt)               # regular setup: load and print networks; create schedulers
# load pre-trained model
model.netG.module.load_state_dict(torch.load('/data/li/Pix2PixNIfTI/checkpoints/liver-98/latest_net_G.pth', map_location=device))
model.netG = model.netG.to(device)
model.netD.module.load_state_dict(torch.load('/data/li/Pix2PixNIfTI/checkpoints/liver-98/latest_net_D.pth', map_location=device))
model.netD = model.netD.to(device)
model.arch_param = torch.load('/data/li/Pix2PixNIfTI/checkpoints/liver-98/arch_parameters.pth', map_location=device)
 
net = model= DenseNet3D()
net = net.to(device=device)
 
##### define optimizer for unet #####
optimizer_densenet = optim.RMSprop(net.parameters(), lr=1e-4, weight_decay=1e-8, momentum=0.9, foreach=True)
scheduler_densenet = optim.lr_scheduler.ReduceLROnPlateau(optimizer_densenet, 'max', patience=5)  # goal: maximize Dice score
# scheduler_unet = optim.lr_scheduler.CosineAnnealingLR(optimizer_unet, T_max=500, eta_min=1e-9)
 
##### prepare dataloader #####
data_loader = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
dataset = data_loader.dataset
total_size = len(dataset)
split_1 = 78
split_2 = 20
split_3 = total_size - split_1 - split_2
 
# Perform the split
subset1, subset2, _ = random_split(dataset, [split_1, split_2, split_3])
 
# Create new DataLoaders
train_loader = DataLoader(subset1, batch_size=opt.batch_size, num_workers=int(opt.num_threads), shuffle=not opt.serial_batches)
val_loader = DataLoader(subset2, batch_size=opt.batch_size, num_workers=int(opt.num_threads), shuffle=not opt.serial_batches)
logging.info('The number of training images = %d' % len(train_loader))
logging.info('The number of validate images = %d' % len(val_loader))
 
train_iters = 5000            # training iterations
total_iters = 0.0              # the total number of training iterations
val_best_score = 0.0           # the best val score of unet
unet_best_score = 0.0          # the best score of unet
 
criterion = nn.BCEWithLogitsLoss()
 
class Generator(ImplicitProblem):
    def training_step(self, batch):
        model.set_input(batch)
        model.forward()
        fake_AB = torch.cat((model.real_A, model.fake_B), 1)
        pred_fake = model.netD(fake_AB)
        loss_G_GAN = model.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        # Compute the L1 loss only on the masked values
        loss_G_L1 = model.criterionL1(model.fake_B * model.mask, model.real_B * model.mask) * model.opt.lambda_L1
        # Compute the L2 loss on the tumor area
        loss_G_L2_T = model.criterionTumor(model.fake_B * model.truth,
                    model.real_B * model.truth) * model.opt.gamma_TMSE
        # print(self.loss_G_L1, self.loss_G_L2_T)
        loss_G_L1 = zero_division(loss_G_L1, torch.sum(model.mask))
        # TODO: Problem what to do with slices without tumor
        loss_G_L2_T = zero_division(loss_G_L2_T, torch.sum(model.truth))
        # print(self.loss_G_L1, self.loss_G_L2_T)
        # combine loss and calculate gradients
        loss_G = loss_G_GAN + loss_G_L1 + loss_G_L2_T
        return loss_G
 
 
class Discriminator(ImplicitProblem):
    def training_step(self, batch):
        model.set_input(batch)
        model.forward()
        fake_AB = torch.cat((model.real_A, model.fake_B), 1)  # we use conditional GANs;
        # we need to feed both input and output to the discriminator
        pred_fake = model.netD(fake_AB.detach())
        loss_D_fake = model.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((model.real_A, model.real_B), 1)
        pred_real = model.netD(real_AB)
        loss_D_real = model.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        return loss_D
 
 
show_index = 0
class DenseNet(ImplicitProblem):
    def training_step(self, batch):
        images = batch['B'].to(device=device, dtype=torch.float32)
        # true_masks = batch['mask'].to(device=device, dtype=torch.long)
        mask_A = batch['A'].to(device=device, dtype=torch.float32)
        labels = batch['label'].to(device=device, dtype=torch.long)
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)
        label_tensor = torch.tensor([labels], dtype=torch.float32)
       
        true_pred = net(images)
        loss = criterion(true_pred, label_tensor)
 
        # fake images and masks
        fake_mask = mask_A
        fake_image = model.netG(fake_mask)
 
        fake_pred = net(fake_image)
        fake_loss = criterion(fake_pred, label_tensor)
       
        densenet_loss = loss + fake_loss
        return densenet_loss
 
 
class Arch(ImplicitProblem):
    def training_step(self, batch):
        # mask_valid = batch['mask'].to(device=device, dtype=torch.long).squeeze(0)
        image_valid = batch['B'].type(torch.cuda.FloatTensor).to(device)
        labels = batch['label'].to(device=device, dtype=torch.long)
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)
        label_tensor = torch.tensor([labels], dtype=torch.float32)
        true_pred = self.densenet(image_valid)
        loss_arch = criterion(true_pred,label_tensor)
        return loss_arch
 
 
class SSEngine(Engine):
 
    @torch.no_grad()
    def validation(self):
        global val_best_score
        val_score = evaluate(self.densenet.module, val_loader, device)
       
        message = 'Performance of DenseNet: '
        message += '%s: %.5f ' % ('densenet_val_score', val_score)
        logging.info(message)
        logger.log({'val_score': val_score})
        if val_score > val_best_score:
            val_best_score = val_score
            torch.save(net.state_dict(), densenet_save_path)
       
        if self.global_step % len(train_loader) == 0 and self.global_step:
            scheduler_densenet.step(val_best_score)            
 
 
outer_config = Config(retain_graph=True)
inner_config = Config(type="darts", unroll_steps=1)
engine_config = EngineConfig(
    valid_step=opt.display_freq * 1,
    train_iters=train_iters,
    roll_back=True,
)
 
netG = Generator(
    name='netG',
    module=model.netG,
    optimizer=model.optimizer_G,
    train_data_loader=train_loader,
    config=inner_config,
    device=device,
)
 
netD = Discriminator(
    name='netD',
    module=model.netD,
    optimizer=model.optimizer_D,
    train_data_loader=train_loader,
    config=inner_config,
    device=device,
)
 
densenet = DenseNet(
    name='densenet',
    module=net,
    optimizer=optimizer_densenet,
    train_data_loader=train_loader,
    config=inner_config,
    device=device,
)
 
optimizer_arch = torch.optim.Adam(arch_parameters(), lr=1e-6, betas=(0.5, 0.999), weight_decay=1e-5)
arch = Arch(
    name='arch',
    module=net,
    optimizer=optimizer_arch,
    train_data_loader=val_loader,
    config=outer_config,
    device=device,
)
 
problems = [netG, netD, densenet, arch]
l2u = {netG: [densenet], densenet: [arch]}
u2l = {arch: [netG]}
# l2u = {}
# u2l = {}
dependencies = {"l2u": l2u, "u2l": u2l}
 
engine = SSEngine(config=engine_config, problems=problems, dependencies=dependencies)
engine.run()
torch.save(net.state_dict(), save_path+'/densenet_final.pkl')