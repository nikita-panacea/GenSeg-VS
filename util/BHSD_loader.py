from pathlib import Path
from os import listdir
from os.path import splitext

from functools import lru_cache

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import cv2


class BHSDDataset(Dataset):
    """Dataset that converts 3-D NIfTI volumes from BHSD into 2-D slice pairs.

    Each sample consists of a single axial slice of the CT volume together with its
    corresponding segmentation mask.  The mask provided in *mask* keeps the original
    integer category (0 = background, 1-5 = bleed sub-types) so that the training
    script can apply *CrossEntropyLoss*.  For the generative branch we also provide
    a *mask_pix2pix* field where all non-zero voxels are collapsed into a single
    foreground channel (binary mask) - this matches the interface used by the
    original ``train_end2end_jsrt.py`` script.
    """

    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = "", clip_min: int = -1024, clip_max: int = 1024):
        super().__init__()
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1.0, "Scale must be within (0, 1]."
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.clip_min = clip_min
        self.clip_max = clip_max

        # Build an index mapping each 2-D slice to its parent volume and slice id
        self.slices = []  # list[tuple[Path, Path, int]] -> (img_path, mask_path, z)
        vol_names = [splitext(f)[0].replace(".nii", "") for f in listdir(self.images_dir) if f.endswith(".nii") or f.endswith(".nii.gz")]
        if not vol_names:
            raise RuntimeError(f"No NIfTI volumes found in {images_dir}.")

        for name in vol_names:
            img_path = self.images_dir / f"{name}.nii.gz"
            if not img_path.exists():
                img_path = self.images_dir / f"{name}.nii"
            mask_path = self.masks_dir / f"{name}{self.mask_suffix}.nii.gz"
            if not mask_path.exists():
                mask_path = self.masks_dir / f"{name}{self.mask_suffix}.nii"
            if not mask_path.exists():
                # Skip volumes without mask
                continue
            # Read header only to get number of slices (cheap)
            z_dim = nib.load(str(img_path)).shape[2]
            for z in range(z_dim):
                self.slices.append((img_path, mask_path, z))
        if not self.slices:
            raise RuntimeError("No matching image/mask pairs were found.")

    @staticmethod
    @lru_cache(maxsize=16)
    def _load_volume(path: Path):
        """Load a NIfTI volume and cache it in memory (up to 16 volumes)."""
        return nib.load(str(path)).get_fdata()

    def __len__(self):
        return len(self.slices)

    def _preprocess_img(self, slice2d: np.ndarray) -> np.ndarray:
        # Clip to soft-tissue window and normalize to [0, 1]
        slice2d = np.clip(slice2d, self.clip_min, self.clip_max)
        slice2d = (slice2d - self.clip_min) / (self.clip_max - self.clip_min)
        slice2d = slice2d.astype(np.float32)
        if self.scale != 1.0:
            h, w = slice2d.shape
            slice2d = cv2.resize(slice2d, (int(w * self.scale), int(h * self.scale)), interpolation=cv2.INTER_LINEAR)
        # Add channel dimension (C, H, W) with C=1
        return slice2d[np.newaxis, ...]

    def _preprocess_mask(self, mask2d: np.ndarray) -> np.ndarray:
        if self.scale != 1.0:
            h, w = mask2d.shape
            mask2d = cv2.resize(mask2d.astype(np.uint8), (int(w * self.scale), int(h * self.scale)), interpolation=cv2.INTER_NEAREST)
        mask2d = mask2d.astype(np.int64)
        return mask2d[np.newaxis, ...]  # keep channel dim for compatibility

    def __getitem__(self, idx: int):
        img_path, mask_path, z = self.slices[idx]
        volume = self._load_volume(img_path)
        mask_vol = self._load_volume(mask_path)

        img_slice = volume[:, :, z]
        mask_slice = mask_vol[:, :, z]

        img = self._preprocess_img(img_slice)
        mask = self._preprocess_mask(mask_slice)
        mask_bin = (mask_slice > 0).astype(np.float32)
        mask_bin = mask_bin[np.newaxis, ...]
        img_tensor = torch.from_numpy(img).float()
        mask_tensor = torch.from_numpy(mask).long()
        mask_pix_tensor = torch.from_numpy(mask_bin).float()

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "image_pix2pix": img_tensor,
            "mask_pix2pix": mask_pix_tensor,
        } 