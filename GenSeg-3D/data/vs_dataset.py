"""Vestibular Schwannoma dataset for classification with GAN augmentation.

Expects a CSV file with columns:
  image_path, segmentation_path, y_value  (y_value in {0,1})

This dataset returns:
  A: source image (INTENSITY)
  B: target image (same as A, since generation augments mask)
  mask: segmentation label map (binary)
  truth: same as mask (used by some losses)
  y_value: class label {0,1}

Transforms mirror those in nifti_dataset (torchio based).
"""
import os
import pandas as pd
import torch
import torchio

from data.base_dataset import BaseDataset, get_params_3d, get_transform_torchio
from util.util import normalize_with_opt


class VSDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--csv_file', type=str, required=True, help='CSV with image_path,segmentation_path,y_value')
        parser.set_defaults(input_nc=1, output_nc=1, preprocess='take_center_and_crop', load_size=64, crop_size=64)
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.df = pd.read_csv(opt.csv_file)
        self.image_paths = self.df['image_path'].tolist()
        self.seg_paths = self.df['segmentation_path'].tolist()
        self.y_values = self.df['y_value'].tolist()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_p = self.image_paths[index]
        seg_p = self.seg_paths[index]
        y = int(self.y_values[index])

        A = torchio.Image(img_p, torchio.INTENSITY)
        B = torchio.Image(img_p, torchio.INTENSITY)  # same image; generator augments via mask
        truth = torchio.LabelMap(seg_p)
        truth.data[truth.data > 0] = 1

        params = get_params_3d(self.opt, A.shape)
        t = get_transform_torchio(self.opt, params)

        A_t = t(A)
        B_t = t(B)
        truth_t = t(truth)
        mask_t = (truth_t.data != 0)

        return {
            'A': A_t.data, 'B': B_t.data,
            'mask': mask_t, 'truth': mask_t,
            'A_paths': img_p, 'B_paths': img_p,
            'y_value': torch.tensor(y, dtype=torch.long),
        }


