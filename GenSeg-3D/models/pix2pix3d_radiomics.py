from typing import Dict, Any

import torch
import torch.nn as nn

from .pix2pix3d_model import Pix2Pix3DModel
from . import networks
from radiomics.features import masked_tensor_stats, features_to_vector, normalize_feature_vector, augment_mask_binary


class Pix2Pix3DRadiomicsModel(Pix2Pix3DModel):
    """
    Pix2Pix3D augmented with a radiomics feature matching loss between
    (real_B, augmented_mask) and (fake_B, augmented_mask).

    Loss: L_total = L_GAN + lambda_L1 * L1(masked) + gamma_TMSE * L2_T + lambda_rad * MSE(f_real, f_fake)
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser = Pix2Pix3DModel.modify_commandline_options(parser, is_train)
        parser.add_argument('--lambda_rad', type=float, default=10.0, help='weight for radiomics feature matching loss')
        parser.add_argument('--mask_aug', type=str, default='dilate', help='mask augmentation for radiomics [dilate|erode|none]')
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        # extend loss names for logging
        if 'G_rad' not in self.loss_names:
            self.loss_names += ['G_rad']

    def _radiomics_mask(self) -> torch.Tensor:
        """Return binary mask for radiomics (optionally augmented)."""
        if self.opt.mask_aug in ('dilate', 'erode'):
            return augment_mask_binary(self.mask, mode=self.opt.mask_aug)
        return self.mask > 0.5

    def backward_G(self):
        # standard pix2pix parts
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        self.loss_G_L1 = self.criterionL1(self.fake_B * self.mask, self.real_B * self.mask) * self.opt.lambda_L1
        self.loss_G_L1 = (self.loss_G_L1 / torch.sum(self.mask).clamp_min(1)).float()

        self.loss_G_L2_T = self.criterionTumor(self.fake_B * self.truth, self.real_B * self.truth) * self.opt.gamma_TMSE
        self.loss_G_L2_T = (self.loss_G_L2_T / torch.sum(self.truth).clamp_min(1)).float()

        # radiomics feature loss: compute feature dicts from tensors so this happens on-device
        with torch.no_grad():
            rmask = self._radiomics_mask()

        feats_real = masked_tensor_stats(self.real_B, rmask)
        feats_fake = masked_tensor_stats(self.fake_B, rmask)

        # Stack to vectors in deterministic (sorted) order
        vec_real = features_to_vector(feats_real)
        vec_fake = features_to_vector(feats_fake)

        # Optionally select subset of features by name (opt.selected_radiomics expects comma-separated names)
        selected = getattr(self.opt, 'selected_radiomics', '')
        if selected:
            names = sorted(feats_real.keys())
            keep = [s.strip() for s in selected.split(',') if s.strip()]
            idx = [i for i, n in enumerate(names) if n in keep]
            if len(idx) == 0:
                # fallback to using all features
                pass
            else:
                vec_real = vec_real[:, idx]
                vec_fake = vec_fake[:, idx]

        # normalize feature vectors (per-batch) to stabilize loss
        # add small epsilon to avoid NaN when std==0
        vec_real = normalize_feature_vector(vec_real).detach()
        vec_fake = normalize_feature_vector(vec_fake)

        # If any NaNs present, replace with zeros
        if torch.isnan(vec_real).any():
            vec_real = torch.nan_to_num(vec_real, nan=0.0, posinf=0.0, neginf=0.0)
        if torch.isnan(vec_fake).any():
            vec_fake = torch.nan_to_num(vec_fake, nan=0.0, posinf=0.0, neginf=0.0)

        # final radiomics MSE
        self.loss_G_rad = nn.functional.mse_loss(vec_fake, vec_real)

        # total
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_L2_T + self.opt.lambda_rad * self.loss_G_rad
        if self.fp16:
            try:
                from apex import amp

                with amp.scale_loss(self.loss_G, self.optimizer_G, loss_id=1) as scaled_loss:
                    scaled_loss.backward()
            except Exception:
                # Apex not available or scaling failed: fallback to normal backward
                self.loss_G.backward()
        else:
            self.loss_G.backward()


