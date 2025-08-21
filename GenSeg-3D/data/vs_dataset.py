import os
import csv
import random
import numpy as np
import nibabel as nib
import torch
from data.base_dataset import BaseDataset


class VSDataset(BaseDataset):
    """Dataset for Vestibular Schwannoma stored as CSV rows with image and mask paths.

    CSV columns expected: patient_id,image_nifti,mask_nifti
    When instantiated from the training script, the script can pass a pre-split
    list of entries via `opt.vs_entries` (list of dicts with keys 'image' and 'mask').
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--vs_csv', type=str, default='', help='path to CSV file listing image and mask paths')
        parser.add_argument('--image_col', type=str, default='image_nifti', help='CSV column name for image path')
        parser.add_argument('--mask_col', type=str, default='mask_nifti', help='CSV column name for mask path')
        parser.add_argument('--train_val_split', type=float, default=0.8, help='train fraction when splitting CSV (if used)')
        parser.add_argument('--use_pyradiomics', action='store_true', help='also extract pyradiomics features and expose them in the sample dict (non-differentiable)')
        parser.add_argument('--selected_radiomics', type=str, default='', help='comma-separated list of radiomics feature names to keep (default: use all)')
        # Avoid re-defining the same option if another module already added it
        if '--mask_aug' not in getattr(parser, '_option_string_actions', {}):
            parser.add_argument('--mask_aug', type=str, default='none', choices=['none', 'dilate', 'erode'], help='mask augmentation mode for radiomics')
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        # opt may carry a pre-built list of entries from an external script
        self.phase = getattr(opt, 'phase', 'train')
        if hasattr(opt, 'vs_entries') and opt.vs_entries is not None:
            self.entries = opt.vs_entries
        else:
            csv_path = getattr(opt, 'vs_csv', '')
            if not csv_path or not os.path.exists(csv_path):
                raise FileNotFoundError(f"VS CSV file not found: {csv_path}")
            self.entries = []
            with open(csv_path, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    img = row.get(opt.image_col) or row.get('image_nifti')
                    msk = row.get(opt.mask_col) or row.get('mask_nifti')
                    if img is None or msk is None:
                        continue
                    self.entries.append({'image': img, 'mask': msk, 'id': row.get('patient_id', '')})

        self.selected_rad = [s.strip() for s in opt.selected_radiomics.split(',')] if getattr(opt, 'selected_radiomics', '') else None
        self.use_pyrad = getattr(opt, 'use_pyradiomics', False)

    def __len__(self):
        return len(self.entries)

    def _load_nifti(self, path):
        nif = nib.load(path)
        arr = nif.get_fdata()
        return arr, nif.affine

    def __getitem__(self, index):
        e = self.entries[index]
        image_path = e['image']
        mask_path = e['mask']

        img, affine = self._load_nifti(image_path)
        msk, _ = self._load_nifti(mask_path)

        # Binarize mask
        msk = (msk != 0).astype(np.uint8)

        # Augmentations that preserve radiomics (applied jointly): random flip and 90deg rotations
        if self.phase == 'train':
            # Random flip along axes
            if random.random() > 0.5:
                axis = random.choice([0, 1, 2])
                img = np.flip(img, axis=axis)
                msk = np.flip(msk, axis=axis)
            # Random rotation by multiples of 90 degrees around (1,2) axes
            k = random.choice([0, 1, 2, 3])
            if k != 0:
                img = np.rot90(img, k=k, axes=(1, 2))
                msk = np.rot90(msk, k=k, axes=(1, 2))

        # Normalize image to 0..1 then to [-1, 1] (pix2pix expected input range)
        if img.max() > img.min():
            img = (img - img.min()) / (img.max() - img.min())
        else:
            img = img * 0.0
        img = img * 2.0 - 1.0

        # Ensure minimum spatial size by center-pad or center-crop to opt.load_size
        target = getattr(self.opt, 'load_size', 64)

        def pad_or_crop_center(vol, target_size):
            # vol: numpy array with shape (D, H, W) or (H, W) etc.
            if vol.ndim != 3:
                return vol
            d, h, w = vol.shape
            td, th, tw = target_size, target_size, target_size
            # crop
            if d >= td:
                startd = (d - td) // 2
                vol = vol[startd:startd + td, :, :]
            else:
                pad_before = (td - d) // 2
                pad_after = td - d - pad_before
                vol = np.pad(vol, ((pad_before, pad_after), (0, 0), (0, 0)), mode='constant')
            d = vol.shape[0]
            if h >= th:
                starth = (h - th) // 2
                vol = vol[:, starth:starth + th, :]
            else:
                pad_before = (th - h) // 2
                pad_after = th - h - pad_before
                vol = np.pad(vol, ((0, 0), (pad_before, pad_after), (0, 0)), mode='constant')
            h = vol.shape[1]
            if w >= tw:
                startw = (w - tw) // 2
                vol = vol[:, :, startw:startw + tw]
            else:
                pad_before = (tw - w) // 2
                pad_after = tw - w - pad_before
                vol = np.pad(vol, ((0, 0), (0, 0), (pad_before, pad_after)), mode='constant')
            return vol

        img = pad_or_crop_center(img, target)
        msk = pad_or_crop_center(msk, target)

        # Convert to torch tensors with shape [1, D, H, W]
        img_t = torch.from_numpy(img.astype('float32')).unsqueeze(0)
        msk_t = torch.from_numpy(msk.astype('uint8')).unsqueeze(0).bool()

        # For compatibility with pix2pix nifti dataset, return keys used by models
        sample = {
            'A': img_t,  # source (we keep image as source)
            'B': img_t,  # target (same image by default - change workflow if you have paired target modality)
            # For radiomics model we use the tumor mask as `mask` so augment_mask_binary operates on tumor region
            'mask': msk_t,
            'truth': msk_t,  # tumor mask used for truth
            'A_paths': image_path,
            'B_paths': image_path,
        }

        # Optionally add pyradiomics features (non-differentiable) into sample for logging or other usage
        if self.use_pyrad:
            try:
                from radiomics.pyrad_extractor import extract_all_features, align_feature_vectors
                feats = extract_all_features(image_path, mask_path)
                vec, names = align_feature_vectors([feats])
                sample['pyrad_vec'] = vec[0]
                sample['pyrad_names'] = names
            except Exception:
                sample['pyrad_vec'] = None
                sample['pyrad_names'] = None

        return sample
