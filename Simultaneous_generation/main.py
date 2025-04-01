from utils.config import parse_args
from utils.data_loader import get_data_loader
from models.gan import GAN
from models.dcgan import DCGAN_MODEL
from models.wgan_clipping import WGAN_CP
from models.wgan_gradient_penalty import WGAN_GP

from deeplab import *
from unet import UNet
from util import util
# from util.JSRT_loader import CarvanaDataset as BasicDataset
# from util.ISIC_loader import CarvanaDataset as BasicDataset
# from util.Breast_loader import CarvanaDataset as BasicDataset
from util.fetoscopy_loader import CarvanaDataset as BasicDataset
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
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.utils import save_image


def main(args):
    device = torch.device('cuda')
    model = None
    if args.model == 'GAN':
        model = GAN(args)
    elif args.model == 'DCGAN':
        model = DCGAN_MODEL(args)
    elif args.model == 'WGAN-CP':
        model = WGAN_CP(args)
    elif args.model == 'WGAN-GP':
        model = WGAN_GP(args)
    else:
        print("Model type non-existing. Try again.")
        exit(-1)
    import pdb; pdb.set_trace()
    ##### Initialize logging #####
    logger = wandb.init(project='WGAN-ISIC', name="unet-40", resume='allow',
                        anonymous='must', mode='disabled')
    logger.config.update(vars(args))
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    ##### prepare dataloader #####
    dataset = BasicDataset(args.dataroot+'/Images_256', args.dataroot+'/Masks_256', 1.0, '_mask')

    n_train = 32 # 165, 35, 9
    n_val = 8
    indices = list(range(len(dataset)))
    train_set = Subset(dataset, indices[:n_train])
    val_set = Subset(dataset, indices[n_train:n_train+n_val])

    loader_args = dict(batch_size=2, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    
    test_set = BasicDataset(args.dataroot.replace('train/', 'test/')+'/Images', args.dataroot.replace('train/', 'test/')+'/Masks', 1.0, '_mask')
    # PH2_dataset = BasicDataset('/data2/li/workspace/data/NLM/Images', '/data2/li/workspace/data/NLM/Masks', 1.0) # use as the out-domain dataset
    # dermIS_dataset = BasicDataset('/data2/li/workspace/data/SZ/Images', '/data2/li/workspace/data/SZ/Masks', 1.0, '_mask') # use as the extra dataset

    # test_loader = DataLoader(test_set, shuffle=False, drop_last=True, **loader_args)
    # test_loader1 = DataLoader(PH2_dataset, shuffle=False, drop_last=True, **loader_args)
    # test_loader2 = DataLoader(dermIS_dataset, shuffle=False, drop_last=True, **loader_args)
    # n_test = 200
    # test_set = Subset(dataset, indices[-n_test:])
    test_loader = DataLoader(test_set, shuffle=False, drop_last=True, **loader_args)

    model.G.load_state_dict(torch.load(os.path.join(os.getcwd(), 'generator_smoke100.pkl')))
    model.G = model.G.cuda()
    # model.train(train_loader)

    # net = UNet(n_channels=3, n_classes=1, bilinear=False)
    net = DeepLabV3(num_classes=1)
    net = net.to(device=device)
    
    # logging.info("#####   Inference   #####")
    # net.load_state_dict(torch.load('./best-kvasir.pth'))
    # JSRT_score = evaluate(net, test_loader, device, True)
    # # NLM_score = evaluate(net, test_loader1, device, True)
    # # SZ_score = evaluate(net, test_loader2, device, True)
    # message = 'Performance on in-domain dataset: '
    # message += '%s: %.5f ' % ('JSRT_score', JSRT_score)
    # # message += 'Performance on out-domain dataset: '
    # # message += '%s: %.5f ' % ('NLM_score', NLM_score)
    # # message += '%s: %.5f ' % ('SZ_score', SZ_score)
    # logging.info(message)
    # sys.exit()
    
    ##### define optimizer for unet #####
    optimizer_unet = optim.RMSprop(net.parameters(), lr=1e-5, 
                                    weight_decay=1e-8, momentum=0.9, foreach=True)
    scheduler_unet = optim.lr_scheduler.ReduceLROnPlateau(optimizer_unet, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=True)

    ##### Training process of UNet #####
    ##### Datasets: train_loader, val_loader, test_loader, PH2_loader, DermIS_loader, all_train_loader #####
    n_epochs = 100
    classes = 1
    train_iters = int(len(train_set) * n_epochs)
    total_iters = 0                # the total number of training iterations
    unet_best_score = 0.0          # the best score of unet
    NLM_best_score = 0.0           # the best score of PH2 dataset
    SZ_best_score = 0.0        # the best score of DermIS dataset

    criterion = nn.CrossEntropyLoss() if classes > 1 else nn.BCEWithLogitsLoss()
    # criterionGAN = networks.GANLoss(opt.gan_mode).to(device)
    # criterionL1 = torch.nn.L1Loss()

    # for i in range(n_epochs):
    while total_iters <= 5000:
        epoch_start_time = time.time()  # timer for entire epoch
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        for i, batch in enumerate(train_loader):  # inner loop within one epoch
            total_iters += 1
            epoch_iter += 1
            
            images = batch['image'].to(device=device, dtype=torch.float32)
            true_masks = batch['mask'].to(device=device, dtype=torch.long)
            
            z = torch.randn(images.shape[0], 100, 1 ,1).to(device=device)
            augmented_data = model.G(z).squeeze()
            
            fake_mask = augmented_data[:, 3,:,:].unsqueeze(1)
            # fake_image = augmented_data[:,0,:,:].unsqueeze(1)
            fake_image = augmented_data[:,:3,:,:]

            # ims_dict = {}
            # ims_dict['fake_mask'] = wandb.Image(fake_mask[0].mul(255).to('cpu', torch.uint8).numpy())
            # ims_dict['fake_image'] = wandb.Image(fake_image[0].mul(255).to('cpu', torch.uint8).numpy().transpose(1,2,0))
            # logger.log(ims_dict)
            
            masks_pred = net(images)
            fake_pred = net(fake_image)
            loss = criterion(masks_pred.squeeze(), true_masks.float().squeeze())
            
            loss += dice_loss(torch.sigmoid(masks_pred.squeeze(1)), true_masks.float().squeeze(1), multiclass=False)
            fake_loss = criterion(fake_pred.squeeze(), fake_mask.float().squeeze())
            fake_loss += dice_loss(torch.sigmoid(fake_pred.squeeze(1)), fake_mask.float().squeeze(1), multiclass=False)
            
            unet_loss = loss + 1.0 * fake_loss

            optimizer_unet.zero_grad(set_to_none=True)
            grad_scaler.scale(unet_loss).backward()
            # torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            grad_scaler.step(optimizer_unet)
            grad_scaler.update()

            if total_iters % 50 == 0:
                test_score = evaluate(net, val_loader, device, True)
                if test_score > unet_best_score:
                    unet_best_score = test_score
                    torch.save(net.state_dict(), './best-smoke50.pth')
                message = 'Performance of UNet: '
                message += '%s: %.5f ' % ('unet_valid_score', test_score)
                message += '%s: %.5f ' % ('unet_best_score', unet_best_score)
                logging.info(message)
                logger.log({'unet_test_score': test_score})

        scheduler_unet.step(unet_best_score)

    logging.info("#####   Inference   #####")
    net.load_state_dict(torch.load('./best-smoke50.pth'))
    JSRT_score = evaluate(net, test_loader, device, True)
    # NLM_score = evaluate(net, test_loader1, device, True)
    # SZ_score = evaluate(net, test_loader2, device, True)
    message = 'Performance on in-domain dataset: '
    message += '%s: %.5f ' % ('JSRT_score', JSRT_score)
    # message += 'Performance on out-domain dataset: '
    # message += '%s: %.5f ' % ('NLM_score', NLM_score)
    # message += '%s: %.5f ' % ('SZ_score', SZ_score)
    logging.info(message)

    '''
    # Start model training
    if args.is_train == 'True':
        train_loader = get_data_loader(args)
        model.train(train_loader)

    # start evaluating on test data
    else:
        model.evaluate(args.load_D, args.load_G)
        # for i in range(50):
        #    model.generate_latent_walk(i)
    '''

if __name__ == '__main__':
    args = parse_args()
    print(args.cuda)
    main(args)
