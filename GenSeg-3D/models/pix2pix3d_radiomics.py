from typing import Dict, Any

import torch
import torch.nn as nn

from .pix2pix3d_model import Pix2Pix3DModel
from . import networks
from radiomics.features import radiomics_feature_vector, augment_mask_binary


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

        # radiomics feature loss
        with torch.no_grad():
            rmask = self._radiomics_mask()
        # Compute features; they are differentiable in volume but mask is treated as constant
        f_real = radiomics_feature_vector(self.real_B, rmask).detach()  # stop grad on target features
        f_fake = radiomics_feature_vector(self.fake_B, rmask)
        self.loss_G_Rad = nn.functional.mse_loss(f_fake, f_real)

        # total
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_L2_T + self.opt.lambda_rad * self.loss_G_Rad
        if self.fp16:
            from apex import amp
            with amp.scale_loss(self.loss_G, self.optimizer_G, loss_id=1) as scaled_loss:
                scaled_loss.backward()
        else:
            self.loss_G.backward()


