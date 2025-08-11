import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def masked_tensor_stats(volume: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Compute simple first-order radiomics-like features on a masked 3D volume.

    Args:
        volume: Tensor of shape [N, 1, D, H, W], float32 in roughly [-1, 1] or normalized range.
        mask:   Tensor of shape [N, 1, D, H, W], bool or {0,1}.

    Returns:
        Dict of feature_name -> Tensor of shape [N]
    """
    if mask.dtype != torch.bool:
        mask_bool = mask > 0.5
    else:
        mask_bool = mask

    eps = 1e-6

    # Clamp volume to finite range
    v = torch.where(mask_bool, volume, torch.zeros_like(volume))
    count = mask_bool.sum(dim=(1, 2, 3, 4)).clamp_min(1).float()

    # First-order stats
    sum_v = v.sum(dim=(1, 2, 3, 4))
    mean = sum_v / count
    # variance
    sq = (v - mean.view(-1, 1, 1, 1, 1)) ** 2
    var = sq.sum(dim=(1, 2, 3, 4)) / count
    std = torch.sqrt(var + eps)
    # min/max over masked voxels: approximate by replacing zeros with +inf/-inf outside mask
    very_neg = torch.finfo(v.dtype).min
    very_pos = torch.finfo(v.dtype).max
    masked_vals = torch.where(mask_bool, v, torch.full_like(v, very_pos))
    v_min = masked_vals.amin(dim=(1, 2, 3, 4))
    masked_vals_max = torch.where(mask_bool, v, torch.full_like(v, very_neg))
    v_max = masked_vals_max.amax(dim=(1, 2, 3, 4))

    # energy (sum of squares)
    energy = (v * v).sum(dim=(1, 2, 3, 4))

    # entropy (histogram-based over masked voxels)
    # Bin to 64 bins across observed range per batch element
    nbins = 64
    # Avoid per-sample loops by using global bins across batch
    gmin = v_min.min()
    gmax = v_max.max()
    # Handle degenerate case
    if torch.isclose(gmax, gmin):
        gmax = gmin + 1.0
    bins = torch.linspace(gmin.item(), gmax.item(), nbins + 1, device=volume.device)
    # Flatten masked voxels per sample
    flat_v = v.view(v.shape[0], -1)
    flat_m = mask_bool.view(mask_bool.shape[0], -1)
    hist_list = []
    for i in range(flat_v.shape[0]):
        vals = flat_v[i][flat_m[i]]
        if vals.numel() == 0:
            hist = torch.zeros(nbins, device=volume.device)
        else:
            # torch.histc is deprecated; use bucketization
            idx = torch.bucketize(vals, bins) - 1  # 0..nbins
            idx = idx.clamp(0, nbins - 1)
            hist = torch.bincount(idx, minlength=nbins).float()
        prob = hist / hist.sum().clamp_min(1.0)
        entropy = -(prob * (prob + eps).log()).sum()
        hist_list.append(entropy)
    entropy = torch.stack(hist_list, dim=0)

    # Simple shape proxy features
    # volume (voxel count) and bounding-box sizes
    vol_vox = count
    # Bounding box per sample
    bb_dims = []
    for i in range(mask_bool.shape[0]):
        m = mask_bool[i, 0]
        if m.any():
            coords = torch.nonzero(m, as_tuple=False).float()
            dmin, hmin, wmin = coords.min(dim=0).values
            dmax, hmax, wmax = coords.max(dim=0).values
            dd = (dmax - dmin + 1.0)
            hh = (hmax - hmin + 1.0)
            ww = (wmax - wmin + 1.0)
        else:
            dd = hh = ww = torch.tensor(0.0, device=volume.device)
        bb_dims.append(torch.stack([dd, hh, ww]))
    bb_dims = torch.stack(bb_dims, dim=0)  # [N, 3]

    return {
        'mean': mean,
        'std': std,
        'min': v_min,
        'max': v_max,
        'energy': energy,
        'entropy': entropy,
        'voxels': vol_vox,
        'bbox_d': bb_dims[:, 0],
        'bbox_h': bb_dims[:, 1],
        'bbox_w': bb_dims[:, 2],
    }


def features_to_vector(features: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Stack feature dict into a [N, F] tensor.
    """
    keys = sorted(features.keys())
    vec = torch.stack([features[k] for k in keys], dim=1)
    return vec


def normalize_feature_vector(vec: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Per-feature standardization across batch to stabilize loss.

    Args:
        vec: [N, F]
    Returns:
        normed: [N, F]
    """
    mean = vec.mean(dim=0, keepdim=True)
    std = vec.std(dim=0, keepdim=True).clamp_min(eps)
    return (vec - mean) / std


def radiomics_feature_vector(volume: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Convenience: compute normalized radiomics vector [N, F] from (volume, mask).
    """
    feats = masked_tensor_stats(volume, mask)
    vec = features_to_vector(feats)
    return normalize_feature_vector(vec)


def augment_mask_binary(mask: torch.Tensor, mode: str = 'dilate', ksize: int = 3) -> torch.Tensor:
    """
    Lightweight binary mask augmentation (3D) without external deps.

    Modes:
      - 'dilate': 3D max-pool on the binary mask
      - 'erode':  3D min-pool implemented via max-pool on inverted mask

    Args:
        mask: [N, 1, D, H, W], bool or {0,1}
    Returns:
        augmented mask of same shape and dtype as input (bool)
    """
    if mask.dtype != torch.bool:
        m = mask > 0.5
    else:
        m = mask

    pad = ksize // 2
    if mode == 'dilate':
        m_f = m.float()
        out = F.max_pool3d(m_f, kernel_size=ksize, stride=1, padding=pad)
        return out > 0.5
    elif mode == 'erode':
        inv = (~m).float()
        out = F.max_pool3d(inv, kernel_size=ksize, stride=1, padding=pad)
        return ~(out > 0.5)
    else:
        return m


