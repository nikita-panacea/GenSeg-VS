#!/usr/bin/env python3
"""Inference and visualization for DenseNet3D classifier.

Usage example (from project root):
  python GenSeg-3D/infer_densenet.py --checkpoint ./checkpoint_e2e/.../densenet.pkl --batch_size 4 --out_dir ./results

This script uses the project's `TrainOptions` to construct the dataset. Pass the same dataset-related
arguments you used during training (e.g. --csv, --dataroot, --dataset_mode, etc.).
"""

import os
import sys
import argparse
import time
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, auc, precision_recall_curve, average_precision_score,
                             confusion_matrix, precision_recall_fscore_support, roc_auc_score)
from torch.utils.data import DataLoader

# ensure project root on path so we can import package modules the same way as training script
sys.path.append('.')

from options.train_options import TrainOptions
from data import create_dataset
from denseNet import DenseNet3D


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def get_label_from_batch(batch):
    # common keys used across datasets: 'label', 'y_value', 'y_label'
    for k in ('label', 'y_value', 'y_label'):
        if k in batch:
            return batch[k]
    # fall back to any tensor-looking entry that's not image tensors
    for k, v in batch.items():
        if k in ('A', 'B', 'mask', 'truth', 'A_paths', 'B_paths'):
            continue
        if isinstance(v, torch.Tensor):
            return v
    raise KeyError('No label field found in batch')


def plot_roc(y_true, y_scores, out_path):
    try:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
    except Exception:
        fpr, tpr, roc_auc = None, None, None

    plt.figure()
    if fpr is not None:
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
    else:
        plt.text(0.5, 0.5, 'ROC not defined (single-class labels)', ha='center')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_pr(y_true, y_scores, out_path):
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)
    except Exception:
        precision, recall, ap = None, None, None

    plt.figure()
    if precision is not None:
        plt.plot(recall, precision, label=f'AP = {ap:.4f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
    else:
        plt.text(0.5, 0.5, 'PR curve not defined (single-class labels)', ha='center')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_confusion(cm, labels, out_path, normalize=False):
    if normalize:
        cm_sum = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, cm_sum, where=(cm_sum != 0))

    plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2. if cm.max() != 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment='center',
                     color='white' if cm[i, j] > thresh else 'black')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    # parse project options for dataset construction
    # NOTE: TrainOptions.parse() calls print_options() which will attempt to create
    # directories under opt.checkpoints_dir. If a user accidentally passes a checkpoint
    # file path into --checkpoints_dir this will raise. To avoid that side-effect we
    # call gather_options() (no printing / mkdir) and then perform minimal parse steps.
    opt = TrainOptions().gather_options()
    opt.isTrain = True
    # set gpu ids (replicating BaseOptions.parse behavior)
    str_ids = opt.gpu_ids.split(',') if hasattr(opt, 'gpu_ids') and opt.gpu_ids is not None else []
    opt.gpu_ids = []
    for str_id in str_ids:
        try:
            id = int(str_id)
        except Exception:
            continue
        if id >= 0:
            opt.gpu_ids.append(id)
    if len(opt.gpu_ids) > 0:
        torch.cuda.set_device(opt.gpu_ids[0])

    # sanitize checkpoints_dir: if user accidentally passed a checkpoint file path
    # into --checkpoints_dir (or if it points to a file), convert it to a directory
    if hasattr(opt, 'checkpoints_dir') and opt.checkpoints_dir:
        # If it's a file path or looks like a checkpoint file, use its parent dir
        if os.path.isfile(opt.checkpoints_dir) or os.path.splitext(opt.checkpoints_dir)[1] in ('.pkl', '.pth'):
            parent = os.path.dirname(opt.checkpoints_dir)
            opt.checkpoints_dir = parent if parent else './checkpoints'

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to densenet checkpoint (.pkl or .pth)')
    parser.add_argument('--batch_size', type=int, default=getattr(opt, 'batch_size', 4))
    parser.add_argument('--num_workers', type=int, default=int(getattr(opt, 'num_threads', 0)))
    parser.add_argument('--out_dir', type=str, default='./inference_results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--threshold', type=float, default=0.5, help='Probability threshold for binary classification')
    args, unknown = parser.parse_known_args()

    # merge unknown args back into opt by naive setattr for dataset args
    # (TrainOptions already consumed CLI; unknown contains unused args)

    device = torch.device(args.device)
    ensure_dir(args.out_dir)

    # build dataset loader using TrainOptions-derived opt
    # ensure batch size / num_workers reflect user's override
    opt.batch_size = args.batch_size
    opt.num_threads = args.num_workers
    dataset_loader = create_dataset(opt)
    dataset = dataset_loader.dataset

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # build model and load weights
    model = DenseNet3D()
    map_location = {'cuda:0': str(device)} if 'cuda' in str(device) else str(device)
    state = torch.load(args.checkpoint, map_location=device)
    # state may be a state_dict or full checkpoint
    try:
        model.load_state_dict(state)
    except Exception:
        # try common nested keys
        if isinstance(state, dict) and 'state_dict' in state:
            model.load_state_dict(state['state_dict'])
        elif isinstance(state, dict) and 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'])
        else:
            raise

    model = model.to(device)
    model.eval()

    all_labels = []
    all_scores = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            # try to obtain input image tensor
            if 'B' in batch:
                images = batch['B'].to(device=device, dtype=torch.float32)
            elif 'A' in batch:
                images = batch['A'].to(device=device, dtype=torch.float32)
            else:
                # pick first tensor that's 4D/5D
                img = None
                for v in batch.values():
                    if isinstance(v, torch.Tensor) and v.dim() >= 4:
                        img = v
                        break
                if img is None:
                    raise RuntimeError('No image tensor found in batch')
                images = img.to(device=device, dtype=torch.float32)

            labels = get_label_from_batch(batch)
            labels = labels.to(device=device)
            if labels.dim() == 1:
                labels = labels.unsqueeze(1)
            labels = labels.float()

            outputs = model(images)
            probs = torch.sigmoid(outputs).detach().cpu().numpy().ravel()
            labels_np = labels.detach().cpu().numpy().ravel()

            all_scores.append(probs)
            all_labels.append(labels_np)

    if len(all_labels) > 0:
        y_true = np.concatenate(all_labels)
        y_scores = np.concatenate(all_scores)
    else:
        y_true = np.array([])
        y_scores = np.array([])

    # binary predictions
    y_pred = (y_scores > args.threshold).astype(int)

    # basic metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    acc = float((y_pred == y_true).sum()) / max(1, y_true.size)
    try:
        roc_auc = float(roc_auc_score(y_true, y_scores)) if y_true.size > 0 and len(np.unique(y_true)) > 1 else None
    except Exception:
        roc_auc = None
    try:
        pr_auc = float(average_precision_score(y_true, y_scores)) if y_true.size > 0 and len(np.unique(y_true)) > 1 else None
    except Exception:
        pr_auc = None

    # save metrics summary
    summary_path = os.path.join(args.out_dir, f'metrics_{int(time.time())}.txt')
    with open(summary_path, 'w') as fh:
        fh.write(f'num_samples: {y_true.size}\n')
        fh.write(f'acc: {acc:.6f}\n')
        fh.write(f'precision: {precision:.6f}\n')
        fh.write(f'recall: {recall:.6f}\n')
        fh.write(f'f1: {f1:.6f}\n')
        fh.write(f'roc_auc: {roc_auc}\n')
        fh.write(f'pr_auc: {pr_auc}\n')

    # save prediction csv
    try:
        import csv
        csv_path = os.path.join(args.out_dir, 'predictions.csv')
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['index', 'label', 'score', 'pred'])
            for i, (lab, scr, pr) in enumerate(zip(y_true.tolist(), y_scores.tolist(), y_pred.tolist())):
                writer.writerow([i, int(lab), float(scr), int(pr)])
    except Exception:
        pass

    # plots
    plot_roc(y_true, y_scores, os.path.join(args.out_dir, 'roc_curve.png'))
    plot_pr(y_true, y_scores, os.path.join(args.out_dir, 'pr_curve.png'))

    try:
        cm = confusion_matrix(y_true, y_pred)
        plot_confusion(cm, labels=['neg', 'pos'], out_path=os.path.join(args.out_dir, 'confusion_matrix.png'), normalize=False)
        plot_confusion(cm, labels=['neg', 'pos'], out_path=os.path.join(args.out_dir, 'confusion_matrix_norm.png'), normalize=True)
    except Exception:
        pass

    print('Inference complete. Results saved to:', args.out_dir)


if __name__ == '__main__':
    main()
