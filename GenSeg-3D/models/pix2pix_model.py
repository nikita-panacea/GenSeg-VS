import torch
from .base_model import BaseModel
from . import networks
from util.util import zero_division
from .networks import arch_parameters


class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
        parser.add_argument('--gamma_TMSE', type=float, default=100.0, help='weight for L2 truth loss in tumor area')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')

        return parser

    def __init__(self, opt, threed=False):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # Print mini-batch shapes once for debugging/inspection
        self._shape_debug_printed = False
        self.mask = None
        self.truth = None
        self.real_A = None
        self.real_B = None
        self.fp16 = opt.fp16
        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'G_L2_T', 'D_real', 'D_fake']
        # specify the images you want to save/display.
        # The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B', 'truth']
        # specify the models you want to save to the disk.
        # The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        self.nifti = True if opt.dataset_mode == "nifti" else False
        # define networks (both generator and discriminator)

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.upsampling,
                                      threed)

        if self.isTrain:  # define a discriminator;
            # conditional GANs need to take both input and output images;
            # Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, threed)

        if self.nifti:
            red_param = "sum"
        else:
            red_param = "mean"
        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss(reduction=red_param)
            self.criterionTumor = torch.nn.MSELoss(reduction=red_param)
            self.arch_param = arch_parameters()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_arch = torch.optim.Adam(self.arch_param, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_arch)

            if len(opt.gpu_ids) > 0 and self.fp16:
                from apex import amp
                [self.netD, self.netG], [self.optimizer_D, self.optimizer_G] = amp.initialize(
                    [self.netD, self.netG],
                    [self.optimizer_D, self.optimizer_G],
                    opt_level='O1',
                    num_losses=2
                )
            if len(opt.gpu_ids) > 0:
                self.netD = torch.nn.DataParallel(self.netD, opt.gpu_ids)  # multi-GPUs
        if len(opt.gpu_ids) > 0:
            self.netG = torch.nn.DataParallel(self.netG, opt.gpu_ids)  # multi-GPUs

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        if self.nifti:
            self.mask = input['mask']
        else:
            # If we are not using nifti the mask is just the image itself
            self.mask = torch.ones(self.real_B.shape, dtype=torch.bool)
        self.mask = self.mask.to(self.device)
        if self.nifti and input['truth'] is not None:
            self.truth = input['truth']
        else:
            # If we don't have the truth, nothing should matter
            self.truth = torch.zeros(self.real_B.shape, dtype=torch.bool)
        self.truth = self.truth.to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        # One-time debug print of incoming mini-batch shapes
        if not self._shape_debug_printed:
            try:
                print(
                    f"[pix2pix] real_A: {tuple(self.real_A.shape)}, real_B: {tuple(self.real_B.shape)}, "
                    f"mask: {tuple(self.mask.shape)}, truth: {tuple(self.truth.shape) if self.truth is not None else None}"
                )
            except Exception:
                pass

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A) A: [1, 1, 64, 64, 64] G(A): [1, 1, 64, 64, 64]
        # One-time debug print of generated mini-batch shape
        if not self._shape_debug_printed:
            try:
                print(f"[pix2pix] fake_B: {tuple(self.fake_B.shape)}")
            except Exception:
                pass
            self._shape_debug_printed = True

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs;
        # we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        if self.fp16:
            from apex import amp
            with amp.scale_loss(self.loss_D, self.optimizer_D, loss_id=0) as scaled_loss:
                scaled_loss.backward()
        else:
            self.loss_D.backward()

    def compute_losses_no_backward(self):
        """Compute and store loss tensors without performing backward.

        Useful for detecting NaNs before applying optimizer steps.
        """
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # conditional GAN
        pred_fake = self.netD(fake_AB.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        loss_D_real = self.criterionGAN(pred_real, True)
        loss_D = (loss_D_fake + loss_D_real) * 0.5

        # Generator parts
        pred_fake_for_G = self.netD(torch.cat((self.real_A, self.fake_B), 1))
        loss_G_GAN = self.criterionGAN(pred_fake_for_G, True)
        loss_G_L1 = self.criterionL1(self.fake_B * self.mask, self.real_B * self.mask) * self.opt.lambda_L1
        loss_G_L1 = zero_division(loss_G_L1, torch.sum(self.mask))
        loss_G_L2_T = self.criterionTumor(self.fake_B * self.truth, self.real_B * self.truth) * self.opt.gamma_TMSE
        loss_G_L2_T = zero_division(loss_G_L2_T, torch.sum(self.truth))

        # Radiomics (use existing helpers if available)
        try:
            from radiomics.features import masked_tensor_stats, features_to_vector, normalize_feature_vector
            with torch.no_grad():
                rmask = self.mask if self.opt.mask_aug == 'none' else None
            feats_real = masked_tensor_stats(self.real_B, rmask)
            feats_fake = masked_tensor_stats(self.fake_B, rmask)
            vec_real = features_to_vector(feats_real)
            vec_fake = features_to_vector(feats_fake)
            vec_real = normalize_feature_vector(vec_real).detach()
            vec_fake = normalize_feature_vector(vec_fake)
            loss_G_rad = nn.functional.mse_loss(vec_fake, vec_real)
        except Exception:
            loss_G_rad = torch.tensor(0.0, device=self.device)

        loss_G = loss_G_GAN + loss_G_L1 + loss_G_L2_T + getattr(self.opt, 'lambda_rad', 0.0) * loss_G_rad

        # Store (but do not backward)
        self.loss_D_fake = loss_D_fake
        self.loss_D_real = loss_D_real
        self.loss_D = loss_D
        self.loss_G_GAN = loss_G_GAN
        self.loss_G_L1 = loss_G_L1
        self.loss_G_L2_T = loss_G_L2_T
        self.loss_G_rad = loss_G_rad
        self.loss_G = loss_G
        return {
            'D_fake': loss_D_fake.item() if hasattr(loss_D_fake, 'item') else float(loss_D_fake),
            'D_real': loss_D_real.item() if hasattr(loss_D_real, 'item') else float(loss_D_real),
            'G_GAN': loss_G_GAN.item() if hasattr(loss_G_GAN, 'item') else float(loss_G_GAN),
            'G_L1': loss_G_L1.item() if hasattr(loss_G_L1, 'item') else float(loss_G_L1),
            'G_L2_T': loss_G_L2_T.item() if hasattr(loss_G_L2_T, 'item') else float(loss_G_L2_T),
            'G_rad': loss_G_rad.item() if hasattr(loss_G_rad, 'item') else float(loss_G_rad),
        }

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        # Compute the L1 loss only on the masked values
        self.loss_G_L1 = self.criterionL1(self.fake_B * self.mask, self.real_B * self.mask) * self.opt.lambda_L1
        # Compute the L2 loss on the tumor area
        self.loss_G_L2_T = self.criterionTumor(self.fake_B * self.truth,
                                               self.real_B * self.truth) * self.opt.gamma_TMSE
        # print(self.loss_G_L1, self.loss_G_L2_T)
        self.loss_G_L1 = zero_division(self.loss_G_L1, torch.sum(self.mask))
        # TODO: Problem what to do with slices without tumor
        self.loss_G_L2_T = zero_division(self.loss_G_L2_T, torch.sum(self.truth))
        # print(self.loss_G_L1, self.loss_G_L2_T)
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_L2_T
        if self.fp16:
            from apex import amp
            with amp.scale_loss(self.loss_G, self.optimizer_G, loss_id=1) as scaled_loss:
                scaled_loss.backward()
        else:
            self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        self.backward_D()  # calculate gradients for D
        self.optimizer_D.step()  # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.optimizer_arch.zero_grad()
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights
        self.optimizer_arch.step()
