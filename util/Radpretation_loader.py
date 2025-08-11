import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2

class RadpretationDataset(Dataset):
    """
    Dataset for Radpretation 2D PNG images and masks, compatible with BHSD dataloader.
    Expects:
      - images_dir: directory with 2D PNG images (all slices for all volumes)
      - masks_dir: directory with subfolders per volume, each containing subfolders per class, each with PNG masks
    Outputs:
      - image: [1, H, W] float32, normalized to [0,1]
      - mask: [1, H, W] int64, class indices 0-5
      - image_pix2pix: same as image
      - mask_pix2pix: [1, H, W] float32, binary mask (foreground/background)
    """
    # Mapping from Radpretation class folder names to BHSD-compatible class indices
    CLASS_MAP = {
        'BackGround': 0,
        'Bleed-Epidural': 1,
        'Bleed-Contusion': 2,  # merged with Bleed-Hematoma below
        'Bleed-Hematoma': 2,   # merged with Bleed-Contusion
        'Bleed-Intraventricular': 3,
        'Bleed-Subarachnoid': 4,
        'Bleed-Subdural': 5,
        # The following are ignored or merged:
        'Bleed-Others': None,
        'Scalp-Hematoma': None,
    }
    N_CLASSES = 6

    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0):
        super().__init__()
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1.0, "Scale must be within (0, 1]."
        self.scale = scale

        # Build index: list of (image_path, mask_paths_dict) for each slice
        self.samples = []
        # Each volume is a subfolder in masks_dir
        for volume_name in sorted(os.listdir(self.masks_dir)):
            volume_mask_dir = self.masks_dir / volume_name
            if not volume_mask_dir.is_dir():
                continue
            # For each class, get list of mask PNGs
            class_to_files = {}
            for class_name in os.listdir(volume_mask_dir):
                class_dir = volume_mask_dir / class_name
                if not class_dir.is_dir() or class_name not in self.CLASS_MAP:
                    continue
                mapped_class = self.CLASS_MAP[class_name]
                if mapped_class is None:
                    continue
                class_to_files.setdefault(mapped_class, []).extend(sorted(os.listdir(class_dir)))
            # Assume all classes have the same slices (by filename)
            # Use the union of all slice filenames across classes
            all_slice_names = set()
            for files in class_to_files.values():
                all_slice_names.update(files)
            for slice_name in sorted(all_slice_names):
                # Find corresponding image in images_dir
                image_path = self.images_dir / slice_name
                if not image_path.exists():
                    continue  # skip if image missing
                # For each class, get mask path if exists
                mask_paths = {}
                for class_idx, files in class_to_files.items():
                    if slice_name in files:
                        mask_paths[class_idx] = volume_mask_dir / [k for k,v in self.CLASS_MAP.items() if v==class_idx][0] / slice_name
                self.samples.append((image_path, mask_paths))
        if not self.samples:
            raise RuntimeError(f"No image/mask pairs found in {images_dir} and {masks_dir}.")

    def __len__(self):
        return len(self.samples)

    def _preprocess_img(self, img: np.ndarray) -> np.ndarray:
        if self.scale != 1.0:
            h, w = img.shape
            img = cv2.resize(img, (int(w * self.scale), int(h * self.scale)), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        return img[np.newaxis, ...]  # [1, H, W]

    def _preprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        if self.scale != 1.0:
            h, w = mask.shape
            mask = cv2.resize(mask.astype(np.uint8), (int(w * self.scale), int(h * self.scale)), interpolation=cv2.INTER_NEAREST)
        return mask[np.newaxis, ...]  # [1, H, W]

    def __getitem__(self, idx: int):
        image_path, mask_paths = self.samples[idx]
        img = np.array(Image.open(image_path).convert('L'))  # grayscale
        mask = np.zeros(img.shape, dtype=np.int64)
        # For each class, add mask if exists
        for class_idx in range(self.N_CLASSES):
            mask_path = mask_paths.get(class_idx, None)
            if mask_path is not None and mask_path.exists():
                mask_img = np.array(Image.open(mask_path).convert('L'))
                mask[mask_img > 0] = class_idx
        img = self._preprocess_img(img)
        mask = self._preprocess_mask(mask)
        mask_bin = (mask > 0).astype(np.float32)
        img_tensor = torch.from_numpy(img).float()
        mask_tensor = torch.from_numpy(mask).long()
        mask_pix_tensor = torch.from_numpy(mask_bin).float()
        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "image_pix2pix": img_tensor,
            "mask_pix2pix": mask_pix_tensor,
        } 