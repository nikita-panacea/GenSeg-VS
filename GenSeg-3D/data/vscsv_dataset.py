"""Alternate VS dataset that reads CSV with header:
Patient_ID, image_nifti, mask_nifti, y_value
"""
import os
import pandas as pd
import torch
import torchio
import numpy as np

from data.base_dataset import BaseDataset, get_params_3d, get_params, get_transform_torchio, get_transform
from util.util import nifti_to_np, normalize_with_opt, np_to_pil


class VscsvDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--csv_file', type=str, required=True, help='CSV with Patient_ID,image_nifti,mask_nifti,y_value')
        parser.add_argument('--chosen_slice', type=int, default=76, help='the slice to choose (in case of 2d)')
        parser.add_argument('--mapping_source', type=str, default="t1", const="t1", nargs='?',
                            choices=['t1', 't2', 't1ce', 'flair'],
                            help='the source sequencing for the mapping')
        parser.add_argument('--mapping_target', type=str, default="t2", const="t2", nargs='?',
                            choices=['t1', 't2', 't1ce', 'flair'],
                            help='the source sequencing for the mapping')
        parser.add_argument('--excel', action='store_true',
                            help='choose to print an excel file with useful information (1) or not (0)')
        parser.add_argument('--smoothing', type=str, default="median", const="median", nargs='?',
                            choices=['average', 'median'],
                            help='the kind of smoothing to apply to the image after mapping')
        parser.add_argument('--show_plots', action='store_true',
                            help='choose to show the final plots for the fake images while testing')
        parser.add_argument('--truth_folder', type=str, default="truth",
                            help='folder where the truth files are saved (if exists).')
        parser.add_argument('--postprocess', type=int, default=-1, const=-1, nargs='?',
                            choices=[-1, 0, 1],
                            help='the kind of post-processing to apply to the images. -1 means no postprocessing, '
                                 '0 means normalize in range [0, 1], '
                                 '1 means normalize with unit variance and mean 0.')
        parser.set_defaults(input_nc=1, output_nc=1, preprocess='take_center_and_crop', load_size=64, crop_size=64)
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        df = pd.read_csv(opt.csv_file)
        self.image_paths = df['image_nifti'].tolist()
        self.segmentation_paths = df['mask_nifti'].tolist()
        self.y_values = df['y_value'].tolist()

        # default behaviour: determine sliced vs 3D based on model
        if opt.model == "pix2pix3d":
            self.sliced = False
        elif opt.model == "pix2pix":
            self.sliced = True
        else:
            # default to 3D behavior
            self.sliced = False

        self.affine = None
        self.original_shape = None
        self.chosen_slice = opt.chosen_slice

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        chosen_imgA = self.image_paths[index]
        chosen_seg = self.segmentation_paths[index]
        y_value = int(self.y_values[index])

        truth = None

        if self.sliced:
            # 2D - extract a single slice from the nifti files
            A, affine = nifti_to_np(chosen_imgA, True, self.chosen_slice)
            seg, _ = nifti_to_np(chosen_seg, True, self.chosen_slice)
            self.original_shape = A.shape
            A = normalize_with_opt(A, 0)
            B = normalize_with_opt(A, 0)  # for this task A and B are the same image
            A = np_to_pil(A)
            B = np_to_pil(B)
            seg = (seg != seg.min()).astype(np.uint8)
            seg = np_to_pil(seg)

            transform_params = get_params(self.opt, A.size)
            c_transform = get_transform(self.opt, transform_params, grayscale=True)

            A_torch = c_transform(A)
            B_torch = c_transform(B)
            seg_t = c_transform(seg)

            truth_torch = (seg_t != seg_t.min())
            A_mask = (A_torch != A_torch.min())

            return {'A': A_torch, 'B': B_torch,
                    'mask': A_mask, 'truth': truth_torch,
                    'A_paths': chosen_imgA, 'B_paths': chosen_seg}
        else:
            # 3D branch using TorchIO
            A = torchio.Image(chosen_imgA, torchio.INTENSITY)
            B = torchio.Image(chosen_imgA, torchio.INTENSITY)  # mapping target is same image
            if os.path.exists(chosen_seg):
                truth = torchio.LabelMap(chosen_seg)
                truth.data[truth.data > 1] = 1

            self.original_shape = A.shape[1:]
            affine = A.affine
            transform_params = get_params_3d(self.opt, A.shape)
            c_transform = get_transform_torchio(self.opt, transform_params)

            A_torch = c_transform(A)
            B_torch = c_transform(B)

            if truth is not None:
                truth = c_transform(truth)
                truth_torch = (truth.data != truth.data.min())
            else:
                truth_torch = torch.zeros(B_torch.data.shape, dtype=torch.bool)

            A_mask = (A_torch.data != A_torch.data.min())
            return {'A': A_torch.data, 'B': B_torch.data,
                    'mask': A_mask, 'truth': truth_torch,
                    'A_paths': chosen_imgA, 'B_paths': chosen_seg,
                    'y_value': torch.tensor(y_value, dtype=torch.long)}

"""Shim to register dataset_mode 'vscsv' with the loader.
This exposes class VscsvDataset defined in vs_dataset_csv.py
under module name data.vscsv_dataset, as expected by the framework.
"""
from .vs_dataset_csv import VscsvDataset


