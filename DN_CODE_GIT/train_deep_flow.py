import os
import h5py as h5
import logging
import math
import numpy as np
from torch.utils.data import Dataset
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch
import time
import mri_unet
from torch.utils.tensorboard import SummaryWriter
import torchsummary
import matplotlib.pyplot as plt
import utils
from block_ops import *

# SSIM, ssim function
from loss_function import *
from data_loading import *

import os
import fnmatch

data_folder = 'Q:/Data8_V2/'


class Flow_Data3D(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, truth_names, case_names, augmentation=False):
        """
        Args:
            truth_names (list): List of paths with folders
            case_names (list): List of paths with folders
            augmentation (bool): Apply augmentation or not
        """

        self.augmentation = augmentation
        self.truth_filenames = truth_names
        self.case_filenames = case_names

    def __len__(self):
        return len(self.truth_filenames)

    def __getitem__(self, idx):
        # Load 3D Images
        # print(f'Load {self.truth_filenames[idx]}')
        truth = load_case(self.truth_filenames[idx])
        # print(f'Load {self.case_filenames[idx]}')
        img = load_case(self.case_filenames[idx])

        img = torch.from_numpy(img)
        truth = torch.from_numpy(truth)

        return img, truth


def run():
    # Get the files
    truth_names = []
    data_names = []
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            if fnmatch.fnmatch(file, 'Acc8*.h5'):
                data_names.append(os.path.join(data_folder, file))
            if fnmatch.fnmatch(file, 'Image*.h5'):
                truth_names.append(os.path.join(data_folder, file))
    print(f'Found {len(truth_names)} cases (truth)')
    print(f'Found {len(data_names)} cases (acc)')

    # Split the data
    Neval = max(int(0.05 * len(truth_names)), 1)
    Ntest = Neval
    Ntrain = len(truth_names) - Ntest - Neval

    # Get a data loader
    test_train = False
    data_train = Flow_Data3D(truth_names[:Ntrain], data_names[:Ntrain], augmentation=True)
    data_eval = Flow_Data3D(truth_names[Ntrain:-Ntest], data_names[Ntrain:-Ntest])
    data_test = Flow_Data3D(truth_names[-Ntest:], data_names[-Ntest:])
    if test_train:
        data_train = Flow_Data3D(truth_names[:5], data_names[:5], augmentation=True)
        data_eval = Flow_Data3D(truth_names[Ntrain:Ntrain + 1], data_names[Ntrain:Ntrain + 1])
        data_test = Flow_Data3D(truth_names[-Ntest:-Ntest + 1], data_names[-Ntest:-Ntest + 1])

    img, truth = data_train[0]
    print(img.shape)
    print(img.dtype)

    mixed_training = True
    mixed_eval = True

    # Creates once at the beginning of training
    if mixed_training:
        scaler = torch.cuda.amp.GradScaler()

    print(f'Found {len(data_train)} slices for training')
    print(f'Found {len(data_eval)} slices for eval')
    print(f'Found {len(data_test)} slices for testing')

    batch_size = 1
    block_size = 100
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size,
                                               shuffle=True, num_workers=2, pin_memory=True)
    eval_loader = torch.utils.data.DataLoader(data_eval, batch_size=1,
                                              shuffle=False, num_workers=0, pin_memory=False)


    # a simple fixed Unet for now
    model = mri_unet.unet.UNet(in_channels=4, out_channels=4, f_maps=96, depth=3, layer_growth=2.0,
                               layer_order=['convolution', 'mod relu'],
                               residual=True, complex_kernel=False, complex_input=True, ndims=3,
                               padding=0)
    #print(model)
    model.cuda()

    # Get the stride
    block_strides = [block_size + model.output_pad[4] + model.output_pad[5],
                    block_size + model.output_pad[2] + model.output_pad[3],
                    block_size + model.output_pad[0] + model.output_pad[1]]
    block_crop = [-model.output_pad[4], -model.output_pad[2], -model.output_pad[0]]

    print(f'Model stride = {block_strides} offset = {block_crop}')

    torchsummary.summary(model, (4, block_size, block_size, block_size), device='cuda', dtypes=[torch.complex64, ])

    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    restart_training = True
    if restart_training:
        #file_old = os.path.join('E:/Deep_Flow/Log/train_8471', 'Flow_Denoiser_316.pt')
        file_old = os.path.join('E:/Deep_Flow/Log/train_218', 'Flow_Denoiser_20.pt')
        state = torch.load(file_old)
        model.load_state_dict(state['state_dict'], strict=True)
        start_epoch = state['epoch'] + 1
        Ntrial = 218
    else:
        start_epoch = 0
        from random import randrange

        Ntrial = randrange(10000)

    local_log_dir = os.path.join(log_dir, f'train_{Ntrial}')
    print(f'Logging to {local_log_dir}')
    writer_train = SummaryWriter(local_log_dir)

        # loss_fcn = nn.MSELoss()
    # loss_fcn = SSIM_loss(val_range=1.0).cuda()
    # loss_fcn = loss_mse
    loss_fcn = angio_weighted_loss
    #loss_fcn = complex_mse

    # loss_fcn = nn.L1Loss()

    device = torch.device("cuda")

    #block_strides = np.array([block_size, block_size, block_size], dtype=int)
    #block_strides = np.array([-model.output_pad[4], -model.output_pad[2], -model.output_pad[0]], dtype=int)
    block_shape = np.array([block_size, block_size, block_size], dtype=int)

    # kern_profile = 1.0/(1.0 + np.exp(2.0/0.25*(np.abs(np.linspace(-1.0,1.0,80))-0.75)))
    kern_profile = np.hamming(block_size)
    kern_profile = kern_profile ** 2
    block_mask = kern_profile.reshape((block_size, 1, 1)) * \
        kern_profile.reshape((1, block_size, 1)) * \
        kern_profile.reshape((1, 1, block_size))
    block_mask = np.reshape(block_mask, (1, 1, block_size, block_size, block_size))
    block_mask = torch.tensor(block_mask)
    block_mask.requires_grad = False

    for epoch in range(start_epoch,500):
        model.train()
        loss_avg = 0.0
        # with torch.autograd.detect_anomaly():
        start_time = time.time()

        for idx, (img3, truth3) in enumerate(train_loader):

            # Scale the images
            scale = 1. / torch.mean(torch.abs(img3))
            img3 *= scale
            truth3 *= scale

            # Zero gradient
            optimizer.zero_grad()

            # Cycle through blocks
            # for iter in range(2):
            if True:

                # if iter==0:
                #     # uniform blocks
                #     num_bz = img3.shape[-3] // block_strides[0]
                #     num_by = img3.shape[-2] // block_strides[1]
                #     num_bx = img3.shape[-1] // block_strides[2]
                #     shift = [0, 0, 0]
                # else:

                # Half shifted blocks
                num_bz = (img3.shape[-3] - 2*block_crop[-3]) // block_strides[0]
                num_by = (img3.shape[-2] - 2*block_crop[-2]) // block_strides[1]
                num_bx = (img3.shape[-1] - 2*block_crop[-1]) // block_strides[2]

                valid_size = [img3.shape[-3] - block_crop[-3] - block_crop[-3],
                              img3.shape[-2] - block_crop[-2] - block_crop[-2],
                              img3.shape[-1] - block_crop[-1] - block_crop[-1]]
                #print(f'Valid size = {valid_size} num b {[num_bx, num_by, num_bz]}')

                # shift = block_shape // 2
                shift = [np.random.randint(s) for s in [img3.shape[-3] - 2*block_crop[-3] - num_bz * block_strides[0],
                                                        img3.shape[-2] - 2*block_crop[-2] - num_by * block_strides[1],
                                                        img3.shape[-1] - 2*block_crop[-1] - num_bx * block_strides[2]]]
                num_blocks = num_bx * num_by * num_bz

                skipped_blocks = int(num_blocks*0.75)
                used_blocks = num_blocks - skipped_blocks
                skip_block = np.zeros( (num_blocks,1), dtype=bool)
                skip_block[:skipped_blocks] = True
                skip_block = np.random.permutation(skip_block)

                count = 0
                for bz in range(num_bz):

                    # Flip axis
                    sz = bz * block_strides[0] + shift[0]
                    ez = sz + block_shape[0]

                    for by in range(num_by):
                        sy = by * block_strides[1] + shift[1]
                        ey = sy + block_shape[1]

                        for bx in range(num_bx):
                            sx = bx * block_strides[2] + shift[2]
                            ex = sx + block_shape[2]

                            if skip_block[count]:
                                count = count + 1
                                continue
                            count = count + 1

                            # Decide to flip axes or not
                            flip = []
                            if np.ndarray.item(np.random.choice([False, True], size=1, p=[0.5, 0.5])):
                                flip.append(-1)
                            if np.ndarray.item(np.random.choice([False, True], size=1, p=[0.5, 0.5])):
                                flip.append(-2)
                            if np.ndarray.item(np.random.choice([False, True], size=1, p=[0.5, 0.5])):
                                flip.append(-3)
                            flip = tuple(flip)

                            # Decide to rotate or not
                            nrot1 = np.random.randint(0, 3)
                            nrot2 = np.random.randint(0, 3)

                            # Grab block
                            img = img3[..., sz:ez, sy:ey, sx:ex].clone()
                            truth =truth3[..., sz:ez, sy:ey, sx:ex].clone()

                            # Augment (flip)
                            img = torch.flip(img, dims=flip)
                            truth = torch.flip(truth, dims=flip)

                            # Augment (rotate)
                            img = torch.rot90(torch.rot90(img, nrot1, [-2, -1]), nrot2, [-3, -1])
                            truth = torch.rot90(torch.rot90(truth, nrot1, [-2, -1]), nrot2, [-3, -1])

                            # Pad and move to device
                            img = img.to(device)
                            truth =  F.pad(truth, model.output_pad).to(device)

                            if mixed_training:
                                # Casts operations to mixed precision
                                with torch.cuda.amp.autocast():
                                    im_est = model(img)
                                    loss = loss_fcn(im_est, truth) / used_blocks

                                # Scales the loss, and calls backward()
                                # to create scaled gradients
                                scaler.scale(loss).backward()

                            else:
                                im_est = model(img)
                                loss = loss_fcn(im_est, truth) / used_blocks


                                loss.backward()

                            loss_avg += loss.detach().item()

            if mixed_training:
                # Unscales gradients and calls
                # or skips optimizer.step()
                scaler.step(optimizer)

                # Updates the scale for next iteration
                scaler.update()
            else:
                optimizer.step()

            # if idx % 100 == 1:
            print(f'  {idx} of {len(train_loader)} : '
                  f'Current loss = {loss_avg / (idx + 1)} , '
                  f'{(idx + 1) * batch_size / (time.time() - start_time)} images/s')

        loss_avg /= len(train_loader)

        model.eval()
        with torch.no_grad():
            loss_avg_eval = 0.0
            for idx, (img3, truth3) in enumerate(eval_loader):

                scale = 1. / torch.mean(torch.abs(img3))
                img3 *= scale
                truth3 *= scale

                if idx == 0:
                    estimate = torch.zeros(img3.shape, dtype=img3.dtype, device=torch.device('cpu'))
                    block_count = torch.zeros(img3.shape, dtype=img3.dtype, device=torch.device('cpu'))

                # Cycle through blocks
                loss3 = 0.0
                num_blocks = 0.0
                if idx == 0:
                    num_iter = 1
                else:
                    num_iter = 1

                for iteration in range(num_iter):

                    # Half shifted blocks
                    num_bz = (img3.shape[-3] - 2 * block_crop[-3]) // block_strides[0]
                    num_by = (img3.shape[-2] - 2 * block_crop[-2]) // block_strides[1]
                    num_bx = (img3.shape[-1] - 2 * block_crop[-1]) // block_strides[2]

                    valid_size = [img3.shape[-3] - block_crop[-3] - block_crop[-3],
                                  img3.shape[-2] - block_crop[-2] - block_crop[-2],
                                  img3.shape[-1] - block_crop[-1] - block_crop[-1]]
                    #print(f'Valid size = {valid_size} num b {[num_bx, num_by, num_bz]}')

                    # shift = block_shape // 2
                    shift = [ s // 2 for s in
                             [img3.shape[-3] - 2 * block_crop[-3] - num_bz * block_strides[0],
                              img3.shape[-2] - 2 * block_crop[-2] - num_by * block_strides[1],
                              img3.shape[-1] - 2 * block_crop[-1] - num_bx * block_strides[2]]]

                    for bz in range(num_bz):
                        sz = bz * block_strides[0] + shift[0]
                        ez = sz + block_shape[0]

                        for by in range(num_by):
                            sy = by * block_strides[1] + shift[1]
                            ey = sy + block_shape[1]

                            for bx in range(num_bx):
                                sx = bx * block_strides[2] + shift[2]
                                ex = sx + block_shape[2]

                                img = img3[..., sz:ez, sy:ey, sx:ex].clone().to(device)
                                #truth = truth3[..., sz:ez, sy:ey, sx:ex].clone().to(device)
                                truth = F.pad(truth3[..., sz:ez, sy:ey, sx:ex].clone(), model.output_pad).to(device)

                                if mixed_eval:
                                    with torch.cuda.amp.autocast():
                                        im_est = model(img)
                                        loss = loss_fcn(im_est, truth)
                                        loss3 += loss.detach().item()
                                else:
                                    im_est = model(img)
                                    loss = loss_fcn(im_est, truth)
                                    loss3 += loss.detach().item()

                                num_blocks += 1.0

                                if idx == 0:
                                    # Store the blocks
                                    estimate[..., sz + block_crop[-3]:ez - block_crop[-3],
                                                sy + block_crop[-2]:ey - block_crop[-2],
                                                sx + block_crop[-1]:ex - block_crop[-1]] +=  im_est.detach().cpu()

                                    # sy:ey, sx:ex] += im_est.detach().cpu() #* block_mask
                                    block_count[..., sz + block_crop[-3]:ez - block_crop[-3],
                                                    sy + block_crop[-2]:ey - block_crop[-2],
                                                    sx + block_crop[-1]:ex - block_crop[-1]] += 1.0

                loss_avg_eval += loss3 / num_blocks

                if idx == 0:
                    estimate /= ( block_count + 1e-5)

                    out_file_name = f'idx0_images.h5'
                    out_file_name = os.path.join(local_log_dir, out_file_name)
                    print(f'Log file to {out_file_name}')

                    sl_img = np.squeeze(img3.detach().cpu().numpy())
                    sl_truth = np.squeeze(truth3.detach().cpu().numpy())
                    sl_est = np.squeeze(estimate.detach().cpu().numpy())
                    sl_block = np.squeeze(block_count.detach().cpu().numpy())

                    if os.path.exists(out_file_name):
                        os.remove(out_file_name)

                    with h5.File(out_file_name, 'w') as hf:
                        hf.create_dataset('truth_abs', data=np.squeeze(np.abs(sl_truth)))
                        hf.create_dataset('image_abs', data=np.squeeze(np.abs(sl_img)))
                        hf.create_dataset('estimate_abs', data=np.squeeze(np.abs(sl_est)))
                        hf.create_dataset('truth_phase', data=np.squeeze(np.angle(sl_truth)))
                        hf.create_dataset('image_phase', data=np.squeeze(np.angle(sl_img)))
                        hf.create_dataset('estimate_phase', data=np.squeeze(np.angle(sl_est)))
                        hf.create_dataset('block_count', data=np.squeeze(np.abs(sl_block)))

                    # Append slices to log
                    out_file_name = f'iter_log.h5'
                    out_file_name = os.path.join(local_log_dir, out_file_name)
                    print(f'Log file to {out_file_name}')

                    write_mode = 'w'
                    if os.path.exists(out_file_name):
                        if epoch == 0:
                            os.remove(out_file_name)
                        else:
                            write_mode = 'a'

                    with  h5.File(out_file_name, write_mode) as hf:
                        sl = sl_truth.shape[1] // 2
                        if epoch == 0:
                            hf.create_dataset('abs_truth', data=np.abs(sl_truth[0,sl,...]))
                            hf.create_dataset('abs_input', data=np.abs(sl_img[0,sl,...]))
                        print(f'Saving to abs_estimate{epoch}')
                        hf.create_dataset(f'abs_estimate{epoch}', data=np.abs(sl_est[0, sl, ...]))

        loss_avg_eval /= len(eval_loader)

        print(f'{epoch} Loss = {loss_avg}, Loss eval {loss_avg_eval}')

        writer_train.add_scalar('Loss_Eval', loss_avg_eval, epoch)
        writer_train.add_scalar('Loss', loss_avg, epoch)

        # save models
        state = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(state, os.path.join(local_log_dir, f'Flow_Denoiser_{epoch}.pt'))


if __name__ == '__main__':
    run()
