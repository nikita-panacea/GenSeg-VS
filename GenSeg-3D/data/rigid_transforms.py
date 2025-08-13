from typing import Tuple
import random
import numpy as np
import torch


def center_crop_or_pad(arr: np.ndarray, target: Tuple[int, int, int]) -> np.ndarray:
    """
    Center-crop or pad a 3D array to target (D, H, W) using constant padding (zeros).
    """
    d, h, w = arr.shape
    td, th, tw = target
    # pad
    pd0 = max((td - d) // 2, 0)
    pd1 = max(td - d - pd0, 0)
    ph0 = max((th - h) // 2, 0)
    ph1 = max(th - h - ph0, 0)
    pw0 = max((tw - w) // 2, 0)
    pw1 = max(tw - w - pw0, 0)
    if pd0 or pd1 or ph0 or ph1 or pw0 or pw1:
        arr = np.pad(arr, ((pd0, pd1), (ph0, ph1), (pw0, pw1)), mode='constant')
        d, h, w = arr.shape
    # crop
    sd = max((d - td) // 2, 0)
    sh = max((h - th) // 2, 0)
    sw = max((w - tw) // 2, 0)
    arr = arr[sd:sd + td, sh:sh + th, sw:sw + tw]
    return arr


def random_flip(arr: np.ndarray) -> np.ndarray:
    for axis in (0, 1, 2):
        if random.random() < 0.5:
            arr = np.flip(arr, axis=axis)
    return arr


def random_rot90_z(arr: np.ndarray) -> np.ndarray:
    """Rotate 0/90/180/270 degrees around z-axis (i.e., in-plane HxW)."""
    k = random.randint(0, 3)
    if k:
        # axes (1, 2) correspond to H, W assuming arr shape (D,H,W)
        arr = np.rot90(arr, k=k, axes=(1, 2))
    return arr


def random_integer_translate(arr: np.ndarray, max_shift: Tuple[int, int, int]) -> np.ndarray:
    """
    Integer translation with zero fill; avoids interpolation.
    max_shift: (sd, sh, sw) maximum absolute shifts for each axis.
    """
    sd = random.randint(-max_shift[0], max_shift[0]) if max_shift[0] > 0 else 0
    sh = random.randint(-max_shift[1], max_shift[1]) if max_shift[1] > 0 else 0
    sw = random.randint(-max_shift[2], max_shift[2]) if max_shift[2] > 0 else 0
    d, h, w = arr.shape
    out = np.zeros_like(arr)
    d0 = max(0, sd)
    h0 = max(0, sh)
    w0 = max(0, sw)
    d1 = min(d, d + sd)
    h1 = min(h, h + sh)
    w1 = min(w, w + sw)
    src = arr[d0 - sd:d1 - sd, h0 - sh:h1 - sh, w0 - sw:w1 - sw]
    out[d0:d1, h0:h1, w0:w1] = src
    return out


def apply_rigid_augs(volume: torch.Tensor, mask: torch.Tensor, target_shape: Tuple[int, int, int], max_shift=(0, 0, 0)):
    """
    Apply identical rigid flips, 90-degree rotations around z, integer translations, and center crop/pad.
    volume, mask: [1, D, H, W] tensors; returns tensors with same shape.
    """
    assert volume.ndim == 4 and mask.ndim == 4
    v = volume.squeeze(0).cpu().numpy()
    m = mask.squeeze(0).cpu().numpy().astype(np.uint8)

    # flips
    v = random_flip(v)
    m = random_flip(m)

    # 90-degree rotation (z-axis)
    v = random_rot90_z(v)
    m = random_rot90_z(m)

    # integer translation
    v = random_integer_translate(v, max_shift)
    m = random_integer_translate(m, max_shift)

    # center crop/pad
    v = center_crop_or_pad(v, target_shape)
    m = center_crop_or_pad(m, target_shape)

    v_t = torch.from_numpy(v).unsqueeze(0).to(volume.dtype)
    m_t = torch.from_numpy(m).unsqueeze(0).to(mask.dtype)
    return v_t, m_t


