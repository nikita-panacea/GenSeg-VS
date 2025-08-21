"""Training script for Vestibular Schwannoma using Pix2Pix3D with radiomics loss.

This script is a wrapper that prepares the dataset (CSV-based), splits into train/val,
and invokes existing `train.py` style training loop but customized to collect
radiomics features and pass them into the model where needed.

It creates a TrainOptions-like opt object by reusing existing option classes and
injecting dataset entries.
"""
import os
import argparse
import random
import csv
from collections import namedtuple
import torch
from options.train_options import TrainOptions
from data import create_dataset
from data.vs_dataset import VSDataset
from models import create_model
from util.util import print_timestamped
import numpy as np


def load_csv_entries(csv_path, image_col='image_nifti', mask_col='mask_nifti', id_col='patient_id'):
    entries = []
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            img = row.get(image_col)
            msk = row.get(mask_col)
            pid = row.get(id_col, '')
            if img and msk:
                entries.append({'image': img, 'mask': msk, 'id': pid})
    return entries


def split_entries(entries, train_frac=0.8, seed=42):
    random.Random(seed).shuffle(entries)
    n = int(len(entries) * train_frac)
    return entries[:n], entries[n:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, help='CSV with image and mask columns')
    parser.add_argument('--checkpoints_dir', default='./checkpoints', help='where to save models')
    parser.add_argument('--name', default='vs_experiment', help='experiment name')
    parser.add_argument('--gpu_ids', default='0', help='gpus')
    parser.add_argument('--train_frac', type=float, default=0.8)
    parser.add_argument('--selected_radiomics', type=str, default='', help='comma-separated names to use (empty = all)')
    parser.add_argument('--mask_aug', type=str, default='none', choices=['none', 'dilate', 'erode'])
    parser.add_argument('--use_pyrad', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args, unknown = parser.parse_known_args()

    # Build base TrainOptions and override
    # Remove script-only args (like --csv) from sys.argv so TrainOptions can parse the standard options
    import sys
    cleaned_argv = [sys.argv[0]]
    i = 1
    while i < len(sys.argv):
        a = sys.argv[i]
        # skip '--csv value' or '--csv=value'
        if a == '--csv' or a.startswith('--csv='):
            if a == '--csv':
                i += 2
                continue
            else:
                i += 1
                continue
        cleaned_argv.append(a)
        i += 1
    sys.argv = cleaned_argv

    opt = TrainOptions().parse()
    opt.checkpoints_dir = args.checkpoints_dir
    opt.name = args.name
    # parse gpu ids string into list of ints (same logic as BaseOptions.parse)
    str_ids = str(args.gpu_ids).split(',') if args.gpu_ids is not None else []
    opt.gpu_ids = []
    for str_id in str_ids:
        try:
            _id = int(str_id)
        except Exception:
            continue
        if _id >= 0:
            opt.gpu_ids.append(_id)
    if len(opt.gpu_ids) > 0:
        torch.cuda.set_device(opt.gpu_ids[0])
    opt.vs_entries = None
    opt.use_pyradiomics = args.use_pyrad
    opt.selected_radiomics = args.selected_radiomics
    opt.mask_aug = args.mask_aug

    entries = load_csv_entries(args.csv)
    train_e, val_e = split_entries(entries, train_frac=args.train_frac, seed=args.seed)

    # Create dataset instances manually and wrap them into the data loader interface
    # We will create two datasets and two dataloaders and perform a simplified training loop
    opt.phase = 'train'
    opt.vs_entries = train_e
    train_dataset = VSDataset(opt)

    opt.phase = 'val'
    opt.vs_entries = val_e
    val_dataset = VSDataset(opt)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=int(opt.num_threads))
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=int(opt.num_threads))

    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    model = create_model(opt)
    model.setup(opt)

    total_iters = 0
    import collections
    import torch.nn.functional as F

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        # Training epoch
        model.train()
        train_sums = collections.defaultdict(float)
        train_batches = 0
        for i, data in enumerate(train_loader):
            total_iters += opt.batch_size
            model.set_input(data)

            # Forward and compute losses without performing backward to detect instability
            try:
                model.forward()
                loss_info = model.compute_losses_no_backward()
            except Exception as e:
                print_timestamped(f"Error computing losses on batch {i}: {e}")
                continue

            # check for NaN/Inf in computed losses
            bad = False
            for v in loss_info.values():
                try:
                    if np.isnan(v) or np.isinf(v):
                        bad = True
                        break
                except Exception:
                    # non-scalar entries
                    pass
            if bad:
                print_timestamped(f"NaN/Inf detected in losses on batch {i}, skipping optimizer step")
                continue

            # Safe to apply optimizer steps (mirror optimize_parameters)
            try:
                # update D
                model.set_requires_grad(model.netD, True)
                model.optimizer_D.zero_grad()
                model.backward_D()
                model.optimizer_D.step()

                # update G
                model.set_requires_grad(model.netD, False)
                model.optimizer_G.zero_grad()
                model.optimizer_arch.zero_grad()
                model.backward_G()
                model.optimizer_G.step()
                try:
                    model.optimizer_arch.step()
                except Exception:
                    pass
            except Exception as e:
                print_timestamped(f"Error during optimizer step on batch {i}: {e}")
                continue

            # accumulate printed losses from model
            losses = model.get_current_losses()
            for k, v in losses.items():
                train_sums[k] += v
            train_batches += 1

            if total_iters % opt.save_latest_freq == 0:
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

        # Average training losses
        train_avgs = {k: (train_sums[k] / train_batches if train_batches else 0.0) for k in train_sums}

        # Validation epoch (compute losses without gradient updates)
        model.eval()
        val_sums = collections.defaultdict(float)
        val_batches = 0
        with torch.no_grad():
            for data in val_loader:
                model.set_input(data)
                model.forward()
                # Compute comparable losses using model's criteria
                # GAN loss (generator)
                fake_AB = torch.cat((model.real_A, model.fake_B), 1)
                pred_fake = model.netD(fake_AB)
                loss_G_GAN = model.criterionGAN(pred_fake, True).item()
                # L1 loss on masked region
                loss_G_L1 = model.criterionL1(model.fake_B * model.mask, model.real_B * model.mask).item() * model.opt.lambda_L1
                loss_G_L1 = (loss_G_L1 / torch.sum(model.mask).clamp_min(1)).item()
                # L2 tumor
                loss_G_L2_T = model.criterionTumor(model.fake_B * model.truth, model.real_B * model.truth).item() * model.opt.gamma_TMSE
                loss_G_L2_T = (loss_G_L2_T / torch.sum(model.truth).clamp_min(1)).item()

                # Radiomics loss (recompute same as in model)
                try:
                    rmask = model._radiomics_mask()
                    feats_real = __import__('radiomics.features', fromlist=['masked_tensor_stats']).masked_tensor_stats(model.real_B, rmask)
                    feats_fake = __import__('radiomics.features', fromlist=['masked_tensor_stats']).masked_tensor_stats(model.fake_B, rmask)
                    vec_real = __import__('radiomics.features', fromlist=['features_to_vector']).features_to_vector(feats_real)
                    vec_fake = __import__('radiomics.features', fromlist=['features_to_vector']).features_to_vector(feats_fake)
                    # optional selection
                    selected = getattr(model.opt, 'selected_radiomics', '')
                    if selected:
                        names = sorted(feats_real.keys())
                        keep = [s.strip() for s in selected.split(',') if s.strip()]
                        idx = [i for i, n in enumerate(names) if n in keep]
                        if idx:
                            vec_real = vec_real[:, idx]
                            vec_fake = vec_fake[:, idx]
                    from radiomics.features import normalize_feature_vector
                    vec_real = normalize_feature_vector(vec_real)
                    vec_fake = normalize_feature_vector(vec_fake)
                    loss_G_rad = F.mse_loss(vec_fake, vec_real).item()
                except Exception:
                    loss_G_rad = 0.0

                # Sum up
                val_sums['G_GAN'] += loss_G_GAN
                val_sums['G_L1'] += loss_G_L1
                val_sums['G_L2_T'] += loss_G_L2_T
                val_sums['G_rad'] += loss_G_rad
                val_batches += 1

        val_avgs = {k: (val_sums[k] / val_batches if val_batches else 0.0) for k in val_sums}

        # Print epoch summary
        print_timestamped(f"Epoch {epoch} train_losses: {train_avgs}")
        print_timestamped(f"Epoch {epoch} val_losses: {val_avgs}")

        model.update_learning_rate()
        if epoch % opt.save_epoch_freq == 0:
            model.save_networks('latest')
            model.save_networks(epoch)

    # final save
    model.save_networks('final')
    print_timestamped('Training finished')


if __name__ == '__main__':
    main()


