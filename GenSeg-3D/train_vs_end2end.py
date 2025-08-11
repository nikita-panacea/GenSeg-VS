import os
import sys
sys.path.append('.')
import time
import logging
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from options.train_options import TrainOptions
from data import create_dataset
from models.pix2pix3d_radiomics import Pix2Pix3DRadiomicsModel
from classifier3d.model import Simple3DClassifier
from models.networks import arch_parameters


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_dataset(dataset, splits=(0.8, 0.2)):
    n = len(dataset)
    n_train = int(n * splits[0])
    n_val = n - n_train
    return random_split(dataset, [n_train, n_val])


@torch.no_grad()
def evaluate_classifier(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        x = batch['B'].to(device=device, dtype=torch.float32)
        y = batch['y_value'].to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    model.train()
    return correct / max(total, 1)


def main():
    set_seed(42)
    opt = TrainOptions().parse()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Use custom dataset mode `vs` (registered via file path data.vs_dataset)
    # Expect user to pass: --dataset_mode vs --csv_file path/to/vs.csv --model pix2pix3d_radiomics
    data_loader = create_dataset(opt)
    full_dataset = data_loader.dataset
    train_set, val_set = split_dataset(full_dataset, (0.8, 0.2))
    train_loader = DataLoader(train_set, batch_size=opt.batch_size, num_workers=int(opt.num_threads), shuffle=not opt.serial_batches)
    val_loader = DataLoader(val_set, batch_size=opt.batch_size, num_workers=int(opt.num_threads), shuffle=False)

    # Build models
    # GAN with radiomics loss (discriminator and generator inside)
    gan_model = Pix2Pix3DRadiomicsModel(opt)
    gan_model.setup(opt)

    # 3D classifier
    clf = Simple3DClassifier(in_channels=1, num_classes=2).to(device)
    clf_opt = optim.Adam(clf.parameters(), lr=1e-4)
    clf_criterion = nn.CrossEntropyLoss()
    # Optimizer for architecture parameters (bilevel-style update using classifier val loss)
    arch_opt = optim.Adam(arch_parameters(), lr=1e-5, betas=(0.5, 0.999))

    best_val_acc = 0.0
    total_iters = 0
    train_iters = 10000

    for step in range(train_iters):
        for batch in train_loader:
            total_iters += 1
            # 1) GAN update with radiomics loss
            gan_model.set_input(batch)
            gan_model.optimize_parameters()

            # 2) Generate synthetic sample to augment classifier training
            with torch.no_grad():
                input_A = batch['A'].to(device=device, dtype=torch.float32)
                fake_image = gan_model.netG(input_A)  # synthesize from A
                y = batch['y_value'].to(device)

            # 3) Classifier update on real and synthetic
            clf_opt.zero_grad()
            logits_real = clf(batch['B'].to(device=device, dtype=torch.float32))
            loss_real = clf_criterion(logits_real, y)
            logits_fake = clf(fake_image.to(device=device, dtype=torch.float32))
            loss_fake = clf_criterion(logits_fake, y)
            loss_clf = (loss_real + loss_fake) * 0.5
            loss_clf.backward()
            clf_opt.step()

            if total_iters % opt.display_freq == 0:
                acc = evaluate_classifier(clf, val_loader, device)
                logging.info(f"step {total_iters}: clf_val_acc={acc:.4f}")
                # Use classifier validation loss to update GAN architecture parameters (bilevel-like):
                # One gradient step on arch_parameters using a small validation batch
                try:
                    arch_opt.zero_grad()
                    val_batch = next(iter(val_loader))
                    with torch.enable_grad():
                        A_val = val_batch['A'].to(device=device, dtype=torch.float32)
                        y_val = val_batch['y_value'].to(device)
                        fake_val = gan_model.netG(A_val)
                        logits_val = clf(fake_val)
                        val_loss = clf_criterion(logits_val, y_val)
                        val_loss.backward()
                        arch_opt.step()
                except Exception:
                    pass
                # Heuristic adjustment for feature-matching weight
                if acc > best_val_acc + 1e-4:
                    best_val_acc = acc
                else:
                    # small adjustment to emphasize feature matching
                    gan_model.opt.lambda_rad = float(gan_model.opt.lambda_rad) * 1.05

            if total_iters >= train_iters:
                break
        if total_iters >= train_iters:
            break

    # Save outputs
    os.makedirs('./checkpoints_vs/', exist_ok=True)
    torch.save(clf.state_dict(), './checkpoints_vs/classifier_best.pth')
    # Save arch params from pix2pix
    try:
        from models.networks import arch_parameters
        torch.save(arch_parameters(), os.path.join('./checkpoints_vs/', 'arch_parameters.pth'))
    except Exception:
        pass


if __name__ == '__main__':
    main()


