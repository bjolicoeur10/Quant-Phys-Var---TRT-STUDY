import os
import h5py as h5
import logging
import math
import numpy as np
from torch.utils.data import Dataset
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch
import time
from unet_model import UNet2D
from torch.utils.tensorboard import SummaryWriter
import torchsummary
import matplotlib.pyplot as plt

# SSIM, ssim function
from loss_function import *
from data_loading import *

data_folder = 'E:/Deep_Flow/Data8_V2/'
out_folder = 'Q:/DeepFlowData2/'

import os
import fnmatch

truth_names = []
data_names = []
for root, dirs, files in os.walk(data_folder):
  for file in files:
    if fnmatch.fnmatch(file, 'Acc8*.h5'):
       data_names.append(os.path.join(data_folder,file))
    if fnmatch.fnmatch(file, 'Image*.h5'):
        truth_names.append(os.path.join(data_folder,file))
case_names = truth_names

regen_data = True
if regen_data == True:
    from pathlib import Path

    Path(out_folder).mkdir(parents=True, exist_ok=True)
    d = -1
    for tname, dname in zip(truth_names, data_names):
        print(f'{dname}')

        d = d + 1

        # Complex 64
        img = load_case(dname)
        truth = load_case(tname)

        # Scale truth to bs one
        scale = 1.0 / np.amax(np.abs(truth))
        truth *= scale

        # Scale image
        #scale_img = np.mean(np.conj(img)*truth ) / np.mean(np.conj(img)*img)
        scale_img = scale
        img *= scale_img

        #  Ave an Image
        from PIL import Image
        temp = np.abs(img[0,160,...])
        temp = 255*temp / np.amax(temp)
        im = Image.fromarray(temp)
        if im.mode != 'RGB':
            im = im.convert('RGB')
        im.save(os.path.join(out_folder,f'image_{d:03}.png'))

        print(f'Scale truth by {scale} ')
        print(f'Scale img by {scale_img}')

        # Convert to 2ch
        img = np.stack([np.real(img), np.imag(img)],axis=0)
        truth = np.stack([np.real(truth), np.imag(truth)], axis=0)

        print(f'Loaded {img.shape} from {tname}')

        # Img will 16x2xNzxNyxNx convert to 16 channel
        img = np.reshape(img, (-1,) + img.shape[-3:])
        truth = np.reshape(truth, (-1,) + img.shape[-3:])

        print(f'Scale image by {scale_img}, scale truth {scale}')

        #print(f'New shape = {img.shape}')

        Path(os.path.join(out_folder, f'{d:03}')).mkdir(parents=True, exist_ok=True)

        ns = 1
        for kidx in range(img.shape[-3]-128):

            # Actual index
            k = kidx + 64

            # Input
            sl_img = img[:, k-ns:k+ns+1, :, :]
            sl_img = np.reshape(sl_img, (-1,) + sl_img.shape[-2:])

            # Output
            sl_truth = truth[:, k, :, :]

            out_file_name = f'{d:03}_slice_{k:04}.pt'
            out_file_name = os.path.join(out_folder, f'{d:03}', out_file_name)

            #m = {'imgsl': sl_img, 'truthsl': sl_truth}
            #torch.save(m, out_file_name)

            with h5.File(out_file_name, 'w') as hf:
                hf.create_dataset('truth', data=sl_truth)
                hf.create_dataset('image', data=sl_img)


exit()

class MPnRAGE_Data(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, folders, augmentation=False):
        """
        Args:
            folders (list): List of paths with folders
            augmentation (bool): Apply augmentation or not
        """

        self.augmentation = augmentation

        fnames = []
        for d in folders:
            f = os.listdir(d)
            f = [os.path.join(d, f) for f in f]
            fnames = fnames + f
        self.filenames = fnames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        with h5.File(self.filenames[idx], 'r') as hf:
            truth = np.array(hf['truth'])
            img = np.array(hf['image'])
            #truth = torch.tensor(hf['truth'])
            #img = torch.tensor(hf['image'])

        if self.augmentation:
            FLIP = np.ndarray.item(np.random.choice([False, True], size=1, p=[0.5, 0.5]))
            scale = np.random.random() * (2.0 - 0.5) + 0.5
            ROT = True
            ROLL = True

            img *= scale
            truth *= scale

            # flip
            if FLIP:
                flip_axis = np.ndarray.item(np.random.choice([0, 1], size=1, p=[0.5, 0.5]))
                img = np.flip(img, flip_axis)
                truth = np.flip(truth, flip_axis)

            if ROLL:
                roll_magLR = np.random.randint(-8, 8)
                roll_magUD = np.random.randint(-8, 8)

                img = np.roll(img, (roll_magLR, roll_magUD), (-2, -1))
                truth = np.roll(truth, (roll_magLR, roll_magUD), (-2, -1))

        img = torch.from_numpy(img)
        truth = torch.from_numpy(truth)

        return img, truth


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def run():
    folders_all = os.listdir(train_file_folder)
    folders_all = [os.path.join(train_file_folder, f) for f in folders_all]
    #Ntrain = int(0.9 * len(folders_all))
    Neval = max( int(0.05 * len(folders_all)),1)
    Ntest = Neval
    Ntrain = len(folders_all) - Ntest - Neval

    data_train = MPnRAGE_Data(folders_all[:Ntrain])
    data_eval = MPnRAGE_Data(folders_all[Ntrain:-Ntest])
    data_test = MPnRAGE_Data(folders_all[-Ntest:])

    print(f'Found {len(data_train)} slices for training')
    print(f'Found {len(data_eval)} slices for eval')
    print(f'Found {len(data_test)} slices for testing')

    batch_size = 6
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    eval_loader = torch.utils.data.DataLoader(data_eval, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    # Estimate the range from some of the data
    minval = []
    maxval = []
    for i in range(100):
        _, truth = data_train[i]
        maxval.append(truth.max())
        minval.append(truth.min())
    val_range = 2.0
    print(f'Estimated range = {val_range}')

    from random import randrange
    Ntrial = randrange(10000)
    writer_train = SummaryWriter(os.path.join(log_dir,f'train_{Ntrial}'))

    # a simple fixed Unet for now
    model = UNet2D(in_channels=8, out_channels=8, f_maps=64, depth=4, layer_growth=1.5, layer_order='cr', residual=True)
    model.cuda()
    print(model)

    torchsummary.summary( model, (8,320,320), device='cuda')

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    loss_mse = nn.MSELoss()
    #loss_fcn = SSIM_loss(val_range=1.0).cuda()
    loss_fcn = loss_mse

    # loss_fcn = nn.L1Loss()

    def complex_to_2ch(array):
        shape = array.shape
        array = np.ascontiguousarray(array)
        array = array.view(dtype=array.real.dtype)
        array = array.reshape(shape + (2,))
        return array


    def to_pytorch(array):
        # Starts (16 x slices x nx x ny )
        array = complex_to_2ch(array)

        # now is  (16 x slices x nx x ny x 2 )
        array = np.ascontiguousarray(np.transpose(array, (1, 0, 4, 2, 3)))
        array = np.ascontiguousarray(np.transpose(array, (1, 0, 4, 2, 3)))
        shape = array.shape
        array = np.reshape(array, (shape[0], shape[1] * shape[2]) + shape[3:])

        tensor = torch.from_numpy(array)

        return tensor.cuda()


    def from_pytorch(array):
        im = array.detach().cpu().numpy()
        return im[:, ::2, ...] + 1j * im[:, 1::2, ...]

    device = torch.device("cuda")
    #im_scale = 1 / val_range


    for epoch in range(100):
        model.train()
        loss_avg = 0.0

        start_time = time.time()
        for idx, (img, truth) in enumerate(train_loader):

            img = img.to(device)
            truth =  truth.to(device)

            img = img[..., 32:-32, :]
            truth = truth[..., 32:-32, :]

            im_est = model(img)
            #loss = loss_fcn(im_est, truth) + loss_mse(im_est, truth)
            loss = loss_fcn(im_est, truth)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            loss_avg += loss.detach().item()

            if idx % 100 == 1:
                print(f'  {idx} : Current loss = {loss_avg / (idx+1)} , {(idx+1)*batch_size / (time.time() - start_time)} images/s')

            #print(loss.detach().item())
        loss_avg /= len(train_loader)

        model.eval()

        loss_avg_eval = 0.0
        for idx, (img, truth) in enumerate(eval_loader):

            img = img.to(device)
            truth = truth.to(device)

            img = img[..., 32:-32, :]
            truth = truth[..., 32:-32, :]

            im_est = model(img)
            #loss = loss_fcn(im_est, truth) + loss_mse(im_est, truth)
            loss = loss_fcn(im_est, truth)

            if idx == int(128/batch_size):
                out_file_name = f'test_{epoch}.h5'
                out_file_name = os.path.join(log_dir, out_file_name)

                sl_img = img.detach().cpu().numpy()
                sl_truth = truth.detach().cpu().numpy()
                sl_est = im_est.detach().cpu().numpy()
                with h5.File(out_file_name, 'w') as hf:
                    hf.create_dataset('truth', data=sl_truth)
                    hf.create_dataset('image', data=sl_img)
                    hf.create_dataset('estimate', data=sl_est)

            loss_avg_eval += loss.detach().item()
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
        torch.save(state, os.path.join(log_dir, f'MPRAGE_MSE_Denoiser_{epoch}.pt'))



    def from_pytorch(array):
        channels = array.shape[-3]
        imout = array[..., ::2, :, :] + 1j * array[..., ::2, :, :]
        return imout.detach().cpu().numpy()


if __name__ == '__main__':
    run()



