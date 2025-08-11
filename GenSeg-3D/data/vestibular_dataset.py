
import os
import pandas as pd
import torchio
import torch
import numpy as np
from data.base_dataset import BaseDataset, get_params_3d, get_transform_torchio
from util.util import error, warning, nifti_to_np, normalize_with_opt


class VestibularDataset(BaseDataset):
    """
    A dataset class for Vestibular Schwannoma MRI data.
    It loads MRI images, corresponding masks, and assigns a class ('increasing' or 'decreasing').
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Adds dataset-specific options to the parser."""
        parser.add_argument('--csv_file', type=str, required=True, help='Path to the CSV file containing data paths and labels.')
        parser.set_defaults(input_nc=1, output_nc=1, preprocess='take_center_and_crop', load_size=64, crop_size=64)
        return parser

    def __init__(self, opt):
        """
        Initializes the VestibularDataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.csv_file = opt.csv_file
        self.data_frame = pd.read_csv(self.csv_file)
        
        self.image_paths = self.data_frame['image_path'].tolist()
        self.segmentation_paths = self.data_frame['segmentation_path'].tolist()
        self.y_values = self.data_frame['y_value'].tolist()

        self.sliced = False # Assuming 3D data as per nifti_dataset.py and UNet3D usage
        self.affine = None
        self.original_shape = None

    def __getitem__(self, index):
        """
        Returns a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It contains the image, mask, and class label.
        """
        image_path = self.image_paths[index]
        segmentation_path = self.segmentation_paths[index]
        y_value = self.y_values[index]

        # Load MRI image
        # Assuming nifti_to_np is suitable for loading the image data
        # For 3D images, sliced should be False, and chosen_slice is not used
        image_np, affine = nifti_to_np(image_path, self.sliced, 0) # 0 for chosen_slice is a placeholder
        self.affine = affine
        
        # Load segmentation mask
        segmentation_np, _ = nifti_to_np(segmentation_path, self.sliced, 0)
        
        # Normalize image data
        image_np = normalize_with_opt(image_np, 0) # Normalize to [0, 1]
        
        # Convert to torchio.Image and torchio.LabelMap
        image_torchio = torchio.Image(tensor=image_np[np.newaxis, ...], affine=affine, type=torchio.INTENSITY)
        mask_torchio = torchio.LabelMap(tensor=segmentation_np[np.newaxis, ...], affine=affine)
        
        # Ensure mask values are binary (0 or 1)
        mask_torchio.data[mask_torchio.data > 0] = 1

        self.original_shape = image_torchio.shape[1:]
        transform_params = get_params_3d(self.opt, image_torchio.shape)
        c_transform = get_transform_torchio(self.opt, transform_params)

        transformed_image = c_transform(image_torchio)
        transformed_mask = c_transform(mask_torchio)

        # Convert y_value to a tensor
        class_label = torch.tensor(y_value, dtype=torch.long)

        return {'A': transformed_image.data, 'B': transformed_image.data, # A and B are same for this model
                'mask': transformed_mask.data, 'y_value': class_label,
                'A_paths': image_path, 'B_paths': segmentation_path}

    def __len__(self):
        """Returns the total number of images in the dataset."""
        return len(self.image_paths) 