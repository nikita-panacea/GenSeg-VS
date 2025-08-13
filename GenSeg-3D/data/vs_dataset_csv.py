"""Alternate VS dataset that reads CSV with header:
Patient_ID, Image_path, Segmentation_path, y_label
"""
import pandas as pd
import torch
import torchio

from data.base_dataset import BaseDataset, get_params_3d, get_transform_torchio


class VscsvDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--csv_file', type=str, required=True, help='CSV with Patient_ID,Image_path,Segmentation_path,y_label')
        parser.set_defaults(input_nc=1, output_nc=1, preprocess='take_center_and_crop', load_size=64, crop_size=64)
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        df = pd.read_csv(opt.csv_file)
        self.pids = df['patient_id'].tolist()
        self.image_paths = df['image_nifti'].tolist()
        self.seg_paths = df['mask_nifti'].tolist()
        self.y_values = df['y_value'].tolist()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_p = self.image_paths[index]
        seg_p = self.seg_paths[index]
        y = int(self.y_values[index])
        A = torchio.Image(img_p, torchio.INTENSITY)
        B = torchio.Image(img_p, torchio.INTENSITY)
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


