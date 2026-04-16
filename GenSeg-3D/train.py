"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from models.networks import arch_parameters
from util.visualizer import Visualizer
from util.util import print_timestamped
import time
import torch
from torch.utils.data import DataLoader, random_split, Subset
from util.util import rad_mse
import os

if __name__ == '__main__':
    opt = TrainOptions().parse()  # get training options
    # Since the 3d training is very intense, we don't print
    if opt.model == "pix2pix3d":
        opt.display_id = -1
    data_loader = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset = data_loader.dataset
    total_size = len(dataset)
    split_1 = 98
    split_2 = total_size - split_1
    # Perform the split
    subset1, subset2 = random_split(dataset, [split_1, split_2])
    dataset = DataLoader(subset1, batch_size=opt.batch_size, num_workers=int(opt.num_threads), shuffle=not opt.serial_batches)
    # validation loader from held-out subset2
    val_loader = DataLoader(subset2, batch_size=1, num_workers=0, shuffle=False)
    dataset_size = len(dataset)  # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    # Print network information
    if opt.print_model_info and "pix2pix" in opt.model:
        from torchsummary import summary
        final_size = opt.crop_size if "crop" in opt.preprocess else opt.load_size
        if "3d" in opt.model:
            summary(model.netG, (opt.input_nc, final_size, final_size, final_size))
        else:
            summary(model.netG, (opt.input_nc, final_size, final_size))
    visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots
    total_iters = 0  # the total number of training iterations
    init_time = time.time()
    
    best_metric = float('inf')

    # track last successfully entered epoch (fallback if loop never starts)
    last_epoch = opt.epoch_count - 1

    try:
        for epoch in range(opt.epoch_count,
                           opt.n_epochs + opt.n_epochs_decay + 1):  # outer loop for different epochs;
            # we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
            epoch_start_time = time.time()  # timer for entire epoch
            iter_data_time = time.time()  # timer for data loading per iteration
            epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
            visualizer.reset()  # reset the visualizer: make sure it saves the results to HTML at least once every epoch

            for i, data in enumerate(dataset):  # inner loop within one epoch
                iter_start_time = time.time()  # timer for computation per iteration
                if total_iters % opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time

                total_iters += opt.batch_size
                epoch_iter += opt.batch_size
                model.set_input(data)  # unpack data from dataset and apply preprocessing
                model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

                if total_iters % opt.display_freq == 0:  # display images on visdom and save images to a HTML file
                    save_result = total_iters % opt.update_html_freq == 0
                    model.compute_visuals()
                    visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

                if total_iters % opt.print_freq == 0:  # print training losses and save logging information to the disk
                    losses = model.get_current_losses()
                    t_comp = (time.time() - iter_start_time) / opt.batch_size
                    visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                    if opt.display_id > 0:
                        visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

                if total_iters % opt.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
                    print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                    save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                    model.save_networks(save_suffix)

                iter_data_time = time.time()
            model.update_learning_rate()  # update learning rates in the end of every epoch.
            if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
                print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
                # always save latest
                model.save_networks('latest')

                # Save arch parameters with epoch metadata so arch_parameters.pth is updated regularly
                try:
                    meta = {
                        'arch': arch_parameters(),
                        'last_epoch': epoch,
                        'total_iters': total_iters
                    }
                    torch.save(meta, os.path.join(model.save_dir, "arch_parameters.pth"))
                except Exception as e:
                    print(f"Failed to save arch_parameters.pth at epoch end: {e}")

                # save best if metric improved (assumes lower metric is better)
                try:
                    current_metric = getattr(model, 'metric', None)
                    if current_metric is not None and current_metric < best_metric:
                        best_metric = current_metric
                        print(f"New best metric: {best_metric:.6f} — saving best checkpoint")
                        model.save_networks('best')
                except Exception:
                    pass
            
            # Run validation after each epoch and print metrics
            try:
                val_metrics = {}
                # accumulate
                total = 0
                acc_global_mse = 0.0
                acc_tumour_mse = 0.0
                acc_tumour_dice = 0.0
                acc_rad = 0.0
                acc_d_real = 0.0
                acc_d_fake = 0.0
                n_rad = 0
                n_tum = 0
                with torch.no_grad():
                    for v in val_loader:
                        model.set_input(v)
                        model.forward()
                        # tensors on device
                        real_B = model.real_B
                        fake_B = model.fake_B
                        mask = model.mask
                        truth = model.truth

                        # global MSE on mask
                        mask_f = mask.to(dtype=real_B.dtype)
                        denom = torch.sum(mask_f).item() if torch.sum(mask_f).item() > 0 else float(real_B.nelement())
                        global_mse = torch.sum(((fake_B - real_B) ** 2) * mask_f).item() / denom
                        acc_global_mse += global_mse

                        # tumour MSE (if tumour present)
                        tcount = torch.sum(truth).item()
                        if tcount > 0:
                            tumour_mse = torch.sum(((fake_B - real_B) ** 2) * truth.to(dtype=real_B.dtype)).item() / tcount
                            acc_tumour_mse += tumour_mse
                            n_tum += 1
                            # tumour dice: threshold fake_B within mask as simple proxy
                            thresh = float(fake_B.mean().item())
                            pred_t = ((fake_B * mask_f) > thresh).to(torch.uint8)
                            true_t = truth.to(torch.uint8)
                            inter = (pred_t & true_t).sum().item()
                            p_sum = pred_t.sum().item()
                            t_sum = true_t.sum().item()
                            dice = (2.0 * inter) / (p_sum + t_sum) if (p_sum + t_sum) > 0 else 0.0
                            acc_tumour_dice += dice

                        # radiomics
                        try:
                            if getattr(model, 'rad_fake', None) is not None and getattr(model, 'rad_real', None) is not None:
                                rm = rad_mse(model.rad_fake, model.rad_real)
                                acc_rad += float(rm)
                                n_rad += 1
                        except Exception:
                            pass

                        # discriminator outputs
                        try:
                            real_ab = torch.cat((model.real_A, model.real_B), 1)
                            fake_ab = torch.cat((model.real_A, model.fake_B), 1)
                            pred_real = model.netD(real_ab).mean().item()
                            pred_fake = model.netD(fake_ab).mean().item()
                            acc_d_real += pred_real
                            acc_d_fake += pred_fake
                        except Exception:
                            pass

                        total += 1

                if total > 0:
                    val_metrics['global_mse'] = acc_global_mse / total
                    val_metrics['tumour_mse'] = (acc_tumour_mse / n_tum) if n_tum > 0 else None
                    val_metrics['tumour_dice'] = (acc_tumour_dice / n_tum) if n_tum > 0 else None
                    val_metrics['rad_mse'] = (acc_rad / n_rad) if n_rad > 0 else None
                    val_metrics['d_real'] = acc_d_real / total
                    val_metrics['d_fake'] = acc_d_fake / total
                    print(f"Validation metrics (epoch {epoch}): {val_metrics}")
            except Exception as e:
                print(f"Validation failed: {e}")

            print('End of epoch %d / %d \t Time Taken: %d sec' % (
                epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

            # mark last successful epoch
            last_epoch = epoch
    finally:
        # Ensure we persist at least the latest checkpoint and arch params even if the script crashed or was interrupted
        print("Finalizing: saving latest checkpoint and arch_parameters.pth (safe-save).")
        try:
            if 'model' in locals():
                try:
                    print("Saving latest model checkpoint (safe-save).")
                    model.save_networks('latest')
                except Exception as e:
                    print(f"Failed to save latest model in finally: {e}")

                # Save best if current metric improved (re-check)
                try:
                    current_metric = getattr(model, 'metric', None)
                    if current_metric is not None and current_metric < best_metric:
                        print("Saving best checkpoint (safe-save) due to improved metric.")
                        model.save_networks('best')
                except Exception:
                    pass

                # Save arch parameters with metadata
                try:
                    meta = {
                        'arch': arch_parameters(),
                        'last_epoch': last_epoch,
                        'total_iters': total_iters
                    }
                    torch.save(meta, os.path.join(model.save_dir, "arch_parameters.pth"))
                    print(f"arch_parameters.pth updated with last_epoch={last_epoch}, total_iters={total_iters}")
                except Exception as e:
                    print(f"Failed to save arch_parameters.pth in finally: {e}")
        except Exception as e:
            print(f"Unexpected error in finally block: {e}")

    end_time = round(time.time() - init_time, 3)
    print_timestamped("The training process took " + str(end_time) + "s.")
