import os
import sys
sys.path.append('.')
import time
import shutil
import wandb
import logging
from util.visualizer import Visualizer
import imgaug as ia
import numpy as np
from PIL import Image
from imgaug import augmenters as iaa
from copy import deepcopy
import gc
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score
 
import torch
import torchvision
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
 
from util import util
from util.util import radiomics_features, rad_mse
from UNet3D.config import (
    TRAINING_EPOCH, NUM_CLASSES, IN_CHANNELS, BCE_WEIGHTS, BACKGROUND_AS_CLASS, TRAIN_CUDA
)
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

# memory diagnostics helper
def print_mem(prefix=""):
    try:
        print(f"{prefix} | CUDA memory allocated: {torch.cuda.memory_allocated() / (1024**3):.3f} GB, "
              f"reserved: {torch.cuda.memory_reserved() / (1024**3):.3f} GB")
        print(torch.cuda.memory_summary(limit=10))
    except Exception as e:
        print("print_mem failed:", e)

# ---------------------------
# Memory & determinism tweaks
# ---------------------------
torch.backends.cudnn.benchmark = True  # generally helps performance on fixed-size inputs
# Optional: torch.backends.cudnn.deterministic = True  # if you need reproducibility (may slow down)
# ensure CUDA visible device default if not set externally
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ---------------------------
# Helpers for freeing memory
# ---------------------------
def free_gpu_cache():
    try:
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass

# counter for saved visualizations
# counters and accumulators
display_counter = 0
train_loss_sum = 0.0
train_loss_count = 0
train_correct = 0
train_total = 0

# counter for visual display
display_counter = 0
 
def evaluate(net, dataloader, device):
    """Return (avg_loss, accuracy, precision, recall, f1, roc_auc, pr_auc) over dataloader."""
    net.eval()
    num_val_batches = len(dataloader)
    total_loss = 0.0
    correct = 0
    total = 0

    y_true_list = []
    y_score_list = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            image, label = batch['B'], batch['label']
            if label.dim() == 1:
                label = label.unsqueeze(1)
            label_tensor = label.to(device=device, dtype=torch.float32)

            image = image.to(device=device, dtype=torch.float32)

            label_pred = net(image)
            batch_loss = criterion(label_pred, label_tensor)
            total_loss += float(batch_loss.item())

            probs = torch.sigmoid(label_pred).detach().cpu().numpy().ravel()
            labels_np = label_tensor.detach().cpu().numpy().ravel()

            y_score_list.append(probs)
            y_true_list.append(labels_np)

            preds = (probs > 0.5).astype(float)
            correct += int((preds == labels_np).sum())
            total += int(labels_np.size)

            # free small intermediate memory each iteration
            del image, label, label_tensor, label_pred, batch_loss, probs, preds, labels_np
            free_gpu_cache()

    net.train()
    avg_loss = total_loss / max(num_val_batches, 1)

    # concatenate lists
    if len(y_true_list) > 0:
        y_true = np.concatenate(y_true_list)
        y_scores = np.concatenate(y_score_list)
    else:
        y_true = np.array([])
        y_scores = np.array([])

    accuracy = (correct / total) if total > 0 else 0.0

    # compute precision/recall/f1
    try:
        if y_true.size > 0:
            y_pred = (y_scores > 0.5).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        else:
            precision = recall = f1 = 0.0
    except Exception:
        precision = recall = f1 = 0.0

    # ROC-AUC and PR-AUC
    try:
        roc_auc = float(roc_auc_score(y_true, y_scores)) if y_true.size > 0 and len(np.unique(y_true)) > 1 else None
    except Exception:
        roc_auc = None
    try:
        pr_auc = float(average_precision_score(y_true, y_scores)) if y_true.size > 0 and len(np.unique(y_true)) > 1 else None
    except Exception:
        pr_auc = None

    return avg_loss, accuracy, precision, recall, f1, roc_auc, pr_auc
 
opt = TrainOptions().parse()   # get training options
# config = get_config(opt)
device = torch.device('cuda:0')

# mixed precision control
USE_AMP = bool(getattr(opt, 'fp16', False))
if USE_AMP:
    print("Mixed precision (autocast) enabled (fp16).")
else:
    print("Mixed precision disabled.")

save_path = './checkpoint_e2e/'+'vs-128-model-'+time.strftime("%Y%m%d-%H%M%S")
if not os.path.exists(save_path):
    os.mkdir(save_path)
densenet_save_path = save_path+'/densenet.pkl'  
 
##### Initialize logging #####
# logger = wandb.init(project='end2end-unet-ISIC', name="unet-200", resume='allow', anonymous='must')
logger = wandb.init(project='end2end-vs', name="end2end-vs-128",
                    resume='allow', anonymous='must', mode='disabled')
logger.config.update(vars(opt))
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
# Silence pyradiomics / SimpleITK and filter warnings to reduce log spam during radiomics extraction
logging.getLogger('radiomics').setLevel(logging.ERROR)
logging.getLogger('radiomics.featureextractor').setLevel(logging.ERROR)
logging.getLogger('SimpleITK').setLevel(logging.ERROR)
import warnings
warnings.filterwarnings('ignore')

# create Visualizer instance to save pix2pix images like `train.py`
visualizer = Visualizer(opt)
# how often to save visuals (iterations). Default: 100
visuals_save_freq = int(getattr(opt, 'visuals_save_freq', 100))
 
##### create models: pix2pix, DenseNet #####
model = create_model(opt)      # create a model given opt.model and other options
model.setup(opt)               # regular setup: load and print networks; create schedulers
# load pre-trained model
model.netG.module.load_state_dict(torch.load('/home/ubuntu/Documents/Nafisha/GenSeg-VS/GenSeg-3D/checkpoints/vs_pix2pix3d_128_model/best_net_G.pth', map_location=device))
model.netG = model.netG.to(device)
model.netD.module.load_state_dict(torch.load('/home/ubuntu/Documents/Nafisha/GenSeg-VS/GenSeg-3D/checkpoints/vs_pix2pix3d_128_model/best_net_D.pth', map_location=device))
model.netD = model.netD.to(device)
model.arch_param = torch.load('/home/ubuntu/Documents/Nafisha/GenSeg-VS/GenSeg-3D/checkpoints/vs_pix2pix3d_128_model/arch_parameters.pth', map_location=device)
 
net = DenseNet3D()
net = net.to(device=device)
 
##### define optimizer for unet #####
optimizer_densenet = optim.RMSprop(net.parameters(), lr=1e-4, weight_decay=1e-8, momentum=0.9, foreach=True)
# minimize validation loss for classification
scheduler_densenet = optim.lr_scheduler.ReduceLROnPlateau(optimizer_densenet, mode='min', patience=5)
# scheduler_unet = optim.lr_scheduler.CosineAnnealingLR(optimizer_unet, T_max=500, eta_min=1e-9)
 
##### prepare dataloader #####
data_loader = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
dataset = data_loader.dataset
total_size = len(dataset)
split_1 = 81
split_2 = 21
split_3 = total_size - split_1 - split_2
 
# Perform the split
subset1, subset2, _ = random_split(dataset, [split_1, split_2, split_3])
 
# Create new DataLoaders
# Use small num_workers to avoid extra memory copies if necessary
num_workers = max(0, int(opt.num_threads)) if hasattr(opt, 'num_threads') else 0
train_loader = DataLoader(subset1, batch_size=opt.batch_size, num_workers=num_workers, shuffle=not opt.serial_batches, pin_memory=True)
val_loader = DataLoader(subset2, batch_size=opt.batch_size, num_workers=num_workers, shuffle=not opt.serial_batches, pin_memory=True)
logging.info('The number of training images = %d' % len(train_loader))
logging.info('The number of validate images = %d' % len(val_loader))
 
train_iters = 5000            # training iterations
total_iters = 0.0              # the total number of training iterations
val_best_score = float('inf')           # lower is better (validation loss)
densenet_best_score = float('inf')          # the best score of densenet (loss)
 
criterion = BCEWithLogitsLoss().to(device) if torch.cuda.is_available() and TRAIN_CUDA else BCEWithLogitsLoss() # Check if Cross Entropy Loss or BCEWithLogitsLoss is better
 
# ---------------------------
# ImplicitProblem wrappers
# ---------------------------

class Generator(ImplicitProblem):
    def training_step(self, batch):
        # We wrap heavy forward/loss computations inside autocast if fp16 requested.
        # The betty engine will handle backward; autocast reduces activation memory.
        if USE_AMP:
            ctx = torch.cuda.amp.autocast()
        else:
            # simple context manager fallback
            class _NullCtx:
                def __enter__(self): return None
                def __exit__(self, exc_type, exc, tb): return False
            ctx = _NullCtx()

        with ctx:
            model.set_input(batch)
            model.forward()
            # Build fake_AB for discriminator input
            fake_AB = torch.cat((model.real_A, model.fake_B), 1)
            pred_fake = model.netD(fake_AB)
            loss_G_GAN = model.criterionGAN(pred_fake, True)

            # L1 loss on masked values
            loss_G_L1 = model.criterionL1(model.fake_B * model.mask, model.real_B * model.mask) * model.opt.lambda_L1
            # L2 loss on tumor area
            loss_G_L2_T = model.criterionTumor(model.fake_B * model.truth,
                                               model.real_B * model.truth) * model.opt.gamma_TMSE

            # Prevent division by zero where mask/truth empty
            loss_G_L1 = zero_division(loss_G_L1, torch.sum(model.mask))
            loss_G_L2_T = zero_division(loss_G_L2_T, torch.sum(model.truth))

            # Radiomics features -- these functions expect CPU arrays, so detach+cpu early
            # and free GPU references ASAP
            # Move only what's needed to CPU, avoid keeping GPU tensors longer than necessary
            try:
                mask_np = model.mask.detach().cpu().int().squeeze().numpy()
            except Exception:
                mask_np = None

            try:
                # convert the images required for radiomics to CPU numpy arrays
                fake_B_cpu = model.fake_B.detach().cpu().squeeze().numpy()
                real_B_cpu = model.real_B.detach().cpu().squeeze().numpy()
            except Exception:
                fake_B_cpu = None
                real_B_cpu = None

            # compute radiomics on CPU arrays (silenced logs)
            try:
                rad_fake = radiomics_features(torch.from_numpy(fake_B_cpu).unsqueeze(0).unsqueeze(0) if fake_B_cpu is not None else model.fake_B.detach().cpu(), 
                                              model.truth.detach().cpu())
                rad_real = radiomics_features(torch.from_numpy(real_B_cpu).unsqueeze(0).unsqueeze(0) if real_B_cpu is not None else model.real_B.detach().cpu(), 
                                              model.truth.detach().cpu())
                loss_G_rad = rad_mse(rad_fake, rad_real) * model.opt.gamma_rad
            except Exception:
                loss_G_rad = 0.0

            # combine losses
            loss_G = loss_G_GAN + loss_G_L1 + loss_G_L2_T + loss_G_rad

        # Free CPU/GPU temporaries ASAP
        try:
            # delete large tensors and run GC + empty_cache
            del fake_AB, pred_fake, loss_G_GAN
            del loss_G_L1, loss_G_L2_T
            # careful: don't delete model attributes that are required later by betty engine,
            # but we can delete locally held references
            del fake_B_cpu, real_B_cpu, rad_fake, rad_real
            free_gpu_cache()
        except Exception:
            pass

        # return a loss tensor on the default device (should be a CUDA tensor)
        return loss_G

class Discriminator(ImplicitProblem):
    def training_step(self, batch):
        if USE_AMP:
            ctx = torch.cuda.amp.autocast()
        else:
            class _NullCtx:
                def __enter__(self): return None
                def __exit__(self, exc_type, exc, tb): return False
            ctx = _NullCtx()

        with ctx:
            model.set_input(batch)
            model.forward()
            fake_AB = torch.cat((model.real_A, model.fake_B), 1)
            pred_fake = model.netD(fake_AB.detach())
            loss_D_fake = model.criterionGAN(pred_fake, False)

            real_AB = torch.cat((model.real_A, model.real_B), 1)
            pred_real = model.netD(real_AB)
            loss_D_real = model.criterionGAN(pred_real, True)

            loss_D = (loss_D_fake + loss_D_real) * 0.5

        # free temporaries
        try:
            del fake_AB, pred_fake, real_AB, pred_real, loss_D_fake, loss_D_real
            free_gpu_cache()
        except Exception:
            pass

        return loss_D

class DenseNet(ImplicitProblem):
    def training_step(self, batch):
        # DenseNet training step; we perform forward on the images and fake images
        if USE_AMP:
            ctx = torch.cuda.amp.autocast()
        else:
            class _NullCtx:
                def __enter__(self): return None
                def __exit__(self, exc_type, exc, tb): return False
            ctx = _NullCtx()

        with ctx:
            images = batch['B'].to(device=device, dtype=torch.float32)
            mask_A = batch['A'].to(device=device, dtype=torch.float32)
            labels = batch['label'].to(device=device, dtype=torch.float32)
            if labels.dim() == 1:
                labels = labels.unsqueeze(1)
            label_tensor = labels

            # forward on real images
            label_pred = net(images)
            loss_real = criterion(label_pred, label_tensor)

            # compute fake image via generator (reuse model.netG already on GPU)
            # ensure we don't keep extra copies of tensors longer than necessary
            fake_image = model.netG(mask_A)
            fake_pred = net(fake_image)
            loss_fake = criterion(fake_pred, label_tensor)

            densenet_loss = loss_real + loss_fake

            # Log training loss/accuracy per batch (minimal logging to avoid flooding)
            try:
                probs_real = torch.sigmoid(label_pred)
                preds_real = (probs_real > 0.5).float()
                batch_correct = int((preds_real == label_tensor).sum().item())
                batch_total = int(label_tensor.numel())
                batch_acc = batch_correct / batch_total if batch_total > 0 else 0.0
                # small rate-limit via global display counter
                if hasattr(self, 'train_log_counter'):
                    self.train_log_counter += 1
                else:
                    self.train_log_counter = 1
                if self.train_log_counter % 10 == 0:
                    logger.log({'train_batch_loss': float(densenet_loss.item()), 'train_batch_acc': batch_acc})
                # accumulate epoch-level train metrics
                try:
                    global train_loss_sum, train_loss_count, train_correct, train_total
                    train_loss_sum += float(densenet_loss.item())
                    train_loss_count += 1
                    train_correct += batch_correct
                    train_total += batch_total
                except Exception:
                    pass
            except Exception:
                pass

        # free temporaries that are not needed further
        try:
            del images, mask_A, labels, label_tensor
            del label_pred, fake_image, fake_pred, loss_real, loss_fake
            free_gpu_cache()
        except Exception:
            pass

        return densenet_loss

class Arch(ImplicitProblem):
    def training_step(self, batch):
        # Arch update uses validation batch; keep this light
        if USE_AMP:
            ctx = torch.cuda.amp.autocast()
        else:
            class _NullCtx:
                def __enter__(self): return None
                def __exit__(self, exc_type, exc, tb): return False
            ctx = _NullCtx()

        with ctx:
            image_valid = batch['B'].to(device=device, dtype=torch.float32)
            labels = batch['label'].to(device=device, dtype=torch.float32)
            if labels.dim() == 1:
                labels = labels.unsqueeze(1)
            label_tensor = labels
            label_pred = self.densenet(image_valid)
            loss_arch = criterion(label_pred, label_tensor)

        try:
            del image_valid, labels, label_tensor, label_pred
            free_gpu_cache()
        except Exception:
            pass

        return loss_arch
 
 
class SSEngine(Engine):
 
    @torch.no_grad()
    def validation(self):
        # Evaluate on validation set and log metrics
        val_loss, val_acc, val_prec, val_rec, val_f1, val_rocauc, val_prauc = evaluate(self.densenet.module, val_loader, device)

        message = (
            f"Performance of DenseNet — val_loss: {val_loss:.5f}, val_acc: {val_acc:.4f}, "
            f"prec: {val_prec:.4f}, rec: {val_rec:.4f}, f1: {val_f1:.4f}, roc_auc: {val_rocauc}, pr_auc: {val_prauc}"
        )
        logging.info(message)
        logger.log({
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_precision': val_prec,
            'val_recall': val_rec,
            'val_f1': val_f1,
            'val_roc_auc': val_rocauc,
            'val_pr_auc': val_prauc,
        })

        # Save the best model by lowest validation loss
        global val_best_score
        if val_loss < val_best_score:
            val_best_score = val_loss
            torch.save(net.state_dict(), densenet_save_path)

        # save pix2pix visuals at validation time (rate-limited by visuals_save_freq)
        try:
            if visuals_save_freq > 0 and (self.global_step % visuals_save_freq == 0):
                model.compute_visuals()
                visuals = model.get_current_visuals()
                # save visuals to a separate folder under checkpoints
                save_dir = os.path.join(opt.checkpoints_dir, opt.name, 'saved_visuals')
                util.mkdirs([save_dir])
                step_dir = os.path.join(save_dir, f'step_{self.global_step}')
                util.mkdirs([step_dir])
                # visuals is an OrderedDict of label -> tensor
                for label, im_data in visuals.items():
                    try:
                        im = util.tensor2im(im_data)
                        out_path = os.path.join(step_dir, f"{label}_step{self.global_step}.png")
                        util.save_image(im, out_path)
                    except Exception:
                        pass
                # also update the Visualizer HTML (optional)
                try:
                    visualizer.display_current_results(visuals, int(self.global_step // max(1, len(train_loader))), save_result=True)
                except Exception:
                    pass
        except Exception:
            pass

        # Step scheduler with validation loss
        if self.global_step % len(train_loader) == 0 and self.global_step:
            scheduler_densenet.step(val_loss)

        # Print and log aggregated epoch train metrics, then reset accumulators
        try:
            global train_loss_sum, train_loss_count, train_correct, train_total
            if train_loss_count > 0:
                epoch_train_loss = train_loss_sum / train_loss_count
                epoch_train_acc = (train_correct / train_total) if train_total > 0 else 0.0
                logging.info(f"Epoch aggregated train metrics — loss: {epoch_train_loss:.5f}, acc: {epoch_train_acc:.4f}")
                logger.log({'epoch_train_loss': epoch_train_loss, 'epoch_train_acc': epoch_train_acc})
            # reset accumulators
            train_loss_sum = 0.0
            train_loss_count = 0
            train_correct = 0
            train_total = 0
        except Exception:
            pass
 
 
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
 