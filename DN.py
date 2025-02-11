#%%
import os
import h5py as h5
import logging
import math
import numpy as np
#from torch.utils.data import Dataset
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
#import torchvision.transforms.functional as TF
import torch
import time
#from unet_model import UNet2D
#from torch.utils.tensorboard import SummaryWriter
#import torchsummary
import matplotlib.pyplot as plt
# from mri_unet import *
#from mri_unet.unet import UNet
import mri_unet
from mri_unet.unet import UNet
from flow_processing import *

#import mri_unet

# SSIM, ssim function
from loss_function import *
from data_loading import *

import os
import fnmatch




base_directory = r"/data/data_mrcv/45_DATA_HUMANS/HEAD_NECK/TEST_RETEST_PCVIPR/trtstudyvol6_00832_2023-08-31/00832_00004_pcvpir_flow_int/interleaf0"
base_directory = r"/data/data_mrcv/45_DATA_HUMANS/HEAD_NECK/TEST_RETEST_PCVIPR/phase_test/scan2"
base_directory = r"/data/data_mrcv/45_DATA_HUMANS/HEAD_NECK/TEST_RETEST_PCVIPR/phase_test/intscan/nc/1"

folders = [base_directory, ]
fileoutput = base_directory
fname = 'Images_0_TR_mixed.h5'
fname = 'Images_nc.h5'
fname = 'Images_1_TA_mixed.h5'
nframes = 1
estimate_all = []
img_all1 = []
for folder in folders:
    file = os.path.join(folder, fname)  # Corrected

    print(f'Loading {file}')

    # file = os.path.join(folder, 'NoCorrection.h5')
    IMGTR = []
    with h5.File(file, 'r') as hf:
        for ef in range(nframes):
            IMGALL = []
            for e in range(4):
                dataname = f'Encode_{e:03}_Frame_{ef:03}'
                # print(dataname)
                temp = np.array(hf['Images'][dataname])

                IMGALL.append(np.array(temp['real'] + 1j * temp['imag']))

            img_all = np.stack(IMGALL, axis=0)
            #     img_all = np.expand_dims(img_all, axis=0)
            # ttemp = img_all[0]
            IMGTR.append(img_all)
        imgtr = np.stack(IMGTR, axis=0)
    img_all = imgtr
    # img_all = np.array(hf['IMAGE'])
    # print(f'Loaded image {img_all.shape}')
    del imgtr, IMGALL, IMGTR

    # a simple fixed Unet for now
    model = UNet(in_channels=4, out_channels=4, f_maps=96, depth=3, layer_growth=2.0,
                 layer_order=['convolution', 'mod relu'],
                 residual=True, complex_kernel=False, complex_input=True, ndims=3,
                 padding=0)

    model.cuda()

    model_name = os.path.join('/data/data_mrcv/45_DATA_HUMANS/HEAD_NECK/TEST_RETEST_PCVIPR/DN_CODE/', 'Flow_Denoiser_588.pt')  # E:/Deep_Flow/Log/train_8471

    # print(f'Loading model from {model_name}')
    state = torch.load(model_name)
    # print(state)
    model.load_state_dict(state['state_dict'], strict=True)

    # Now lets run through a case
    # Complex 64
    # count = 666
    img_all_temp = img_all
    for i in range(nframes):
        img_all = img_all_temp[i]
        img_all = np.expand_dims(img_all, axis=0)
        count = i
        for tframe in range(img_all.shape[0]):
            estimate_all = []
            img_all1 = []

            img = img_all[tframe][-4:]  # 4 changed img_all to temp_img_all
            # img = img_all[tframe]
            # truth = truth_all[tframe]
            # print(f'Img shape = {img.shape}')

            # New scaling for time resolved
            scale_img = 1 / np.mean(np.abs(img))  # change this to 1.5 and .5
            # print(f'Scale (img) = {scale_img}')
            # scale_img = 1. / np.amax(np.abs(truth))
            # print(f'Scale (truth) = {scale_img}')

            # Scale image
            # scale_img = np.mean(np.conj(img)*truth ) / np.mean(np.conj(img)*img)
            img *= scale_img
            img = torch.from_numpy(img)

            # print(f'Scale img by {scale_img}')
            img_temp = []
            for fr in range(4):
                nopad = img[fr, :, :, :]
                padded = np.pad(nopad, (20, 20), 'reflect')
                img_temp.append(padded)
            IMG_TEMP = np.stack(img_temp)
            img = IMG_TEMP
            img = torch.from_numpy(img)
            # img = np.expand_dims(IMG_TEMP, axis=0)

            # Setup Blocking
            block_size = 80
            block_strides = [block_size + model.output_pad[4] + model.output_pad[5],
                             block_size + model.output_pad[2] + model.output_pad[3],
                             block_size + model.output_pad[0] + model.output_pad[1]]
            block_crop = [-model.output_pad[4], -model.output_pad[2], -model.output_pad[0]]
            block_shape = np.array([block_size, block_size, block_size], dtype=int)

            estimate = torch.zeros(img.shape, dtype=img.dtype, device=torch.device('cpu'),
                                   requires_grad=False)
            block_count = torch.zeros(img.shape, dtype=img.dtype, device=torch.device('cpu'),
                                      requires_grad=False)

            num_bz = (img.shape[-3] - 2 * block_crop[-3]) // block_strides[0]
            num_by = (img.shape[-2] - 2 * block_crop[-2]) // block_strides[1]
            num_bx = (img.shape[-1] - 2 * block_crop[-1]) // block_strides[2]

            shift = [s // 2 for s in
                     [img.shape[-3] - 2 * block_crop[-3] - num_bz * block_strides[0],
                      img.shape[-2] - 2 * block_crop[-2] - num_by * block_strides[1],
                      img.shape[-1] - 2 * block_crop[-1] - num_bx * block_strides[2]]]

            for bz in range(num_bz):
                sz = bz * block_strides[0] + shift[0]
                ez = sz + block_shape[0]

                for by in range(num_by):
                    sy = by * block_strides[1] + shift[1]
                    ey = sy + block_shape[1]

                    for bx in range(num_bx):
                        sx = bx * block_strides[2] + shift[2]
                        ex = sx + block_shape[2]

                        img_block = img[..., sz:ez, sy:ey, sx:ex].clone().cuda()

                        with torch.no_grad():
                            img_block = img_block.cuda().unsqueeze(0)
                            with torch.cuda.amp.autocast():
                                estimate_block = model(img_block).squeeze(0)

                            estimate[..., sz + block_crop[-3]:ez - block_crop[-3],
                            sy + block_crop[-2]:ey - block_crop[-2],
                            sx + block_crop[-1]:ex - block_crop[-1]] += estimate_block.detach().cpu()
            est_temp = []
            for fra in range(4):
                nopad = estimate[fra, :, :, :]
                unpad = nopad[20:340, 20:340, 20:340]
                est_temp.append(unpad)
            EST_TEMP = np.stack(est_temp)
            estimate = EST_TEMP
            estimate = torch.from_numpy(estimate)

            img_temp = []
            for fra in range(4):
                nopad = img[fra, :, :, :]
                unpad = nopad[20:340, 20:340, 20:340]
                img_temp.append(unpad)
            IMG_TEMP = np.stack(img_temp)
            img = IMG_TEMP
            img = torch.from_numpy(img)

            estimate_all.append(estimate.detach().cpu().numpy())
            img_all1.append(img.detach().cpu().numpy())
            #     # Make sure this is 5D
            img = img.unsqueeze(0).numpy()
            estimate = estimate.unsqueeze(0).numpy()

        estimate = np.stack(estimate_all)
        img = np.stack(img_all1)
        num_encodes = 4
        if num_encodes == 5:
            encoding = "5pt"
        elif num_encodes == 4:
            encoding = "4pt-referenced"
            # encoding = "5pt-cut"
        elif num_encodes == 3:
            encoding = "3pt"
        # print(f' encoding type is {encoding}')

        temp = np.moveaxis(estimate, 1, -1)

        # print(f'Passing {temp.shape} to mri_flow')
        # Solve for Velocity
        mri_flow = MRI_4DFlow(encode_type=encoding, venc=80)
        mri_flow.signal = temp
        mri_flow.solve_for_velocity()
        mri_flow.update_angiogram()
        mri_flow.background_phase_correct()
        mri_flow.update_angiogram()

        # Export to file
        folder = fileoutput
        out_name = os.path.join(folder, 'FlowE.h5')
        try:
            pass
        except OSError:
            pass
        with h5py.File(out_name, 'a') as hf:
            if i == 0:
                grp = hf.create_group('Data')
            else:
                grp = hf['Data']

            if i < 10:
                grp.create_dataset(f"ph_00{i}_vd_1", data=np.squeeze(mri_flow.velocity_estimate[..., 0])*10)
                grp.create_dataset(f"ph_00{i}_vd_2", data=np.squeeze(mri_flow.velocity_estimate[..., 1])*10)
                grp.create_dataset(f"ph_00{i}_vd_3", data=np.squeeze(mri_flow.velocity_estimate[..., 2])*10)
                # grp.create_dataset("VMAG", data=np.squeeze(np.sqrt(np.sum(mri_flow.velocity_estimate ** 2, axis=-1))))
                grp.create_dataset(f"ph_00{i}_cd",
                                   data=np.squeeze(mri_flow.angiogram) / np.max(mri_flow.angiogram))
                grp.create_dataset(f"ph_00{i}_mag",
                                   data=np.squeeze(mri_flow.magnitude) / np.max(mri_flow.magnitude))
            else:
                grp.create_dataset(f"ph_0{i}_vd_1", data=np.squeeze(mri_flow.velocity_estimate[..., 0])*10)
                grp.create_dataset(f"ph_0{i}_vd_2", data=np.squeeze(mri_flow.velocity_estimate[..., 1])*10)
                grp.create_dataset(f"ph_0{i}_vd_3", data=np.squeeze(mri_flow.velocity_estimate[..., 2])*10)
                # grp.create_dataset("VMAG", data=np.squeeze(np.sqrt(np.sum(mri_flow.velocity_estimate ** 2, axis=-1))))
                grp.create_dataset(f"ph_0{i}_cd",
                                   data=np.squeeze(mri_flow.angiogram) / np.max(mri_flow.angiogram))
                grp.create_dataset(f"ph_0{i}_mag",
                                   data=np.squeeze(mri_flow.magnitude) / np.max(mri_flow.magnitude))

        temp = np.moveaxis(img, 1, -1)

        # print(f'Passing {temp.shape} to mri_flow')
        # Solve for Velocity
        mri_flow = MRI_4DFlow(encode_type=encoding, venc=80)
        mri_flow.signal = temp
        mri_flow.solve_for_velocity()
        mri_flow.update_angiogram()
        mri_flow.background_phase_correct()
        mri_flow.update_angiogram()

        out_name = os.path.join(folder, 'FlowS.h5')
        try:
            pass
        except OSError:
            pass
        with h5py.File(out_name, 'a') as hf:
            if i == 0:
                grp = hf.create_group('Data')
            else:
                grp = hf['Data']
            if i < 10:
                grp.create_dataset(f"ph_00{i}_vd_1", data=np.squeeze(mri_flow.velocity_estimate[..., 0])*10)
                grp.create_dataset(f"ph_00{i}_vd_2", data=np.squeeze(mri_flow.velocity_estimate[..., 1])*10)
                grp.create_dataset(f"ph_00{i}_vd_3", data=np.squeeze(mri_flow.velocity_estimate[..., 2])*10)
                # grp.create_dataset("VMAG", data=np.squeeze(np.sqrt(np.sum(mri_flow.velocity_estimate ** 2, axis=-1))))
                grp.create_dataset(f"ph_00{i}_cd",
                                   data=np.squeeze(mri_flow.angiogram) / np.max(mri_flow.angiogram))
                grp.create_dataset(f"ph_00{i}_mag",
                                   data=np.squeeze(mri_flow.magnitude) / np.max(mri_flow.magnitude))
            else:
                grp.create_dataset(f"ph_0{i}_vd_1", data=np.squeeze(mri_flow.velocity_estimate[..., 0])*10)
                grp.create_dataset(f"ph_0{i}_vd_2", data=np.squeeze(mri_flow.velocity_estimate[..., 1])*10)
                grp.create_dataset(f"ph_0{i}_vd_3", data=np.squeeze(mri_flow.velocity_estimate[..., 2])*10)
                # grp.create_dataset("VMAG", data=np.squeeze(np.sqrt(np.sum(mri_flow.velocity_estimate ** 2, axis=-1))))
                grp.create_dataset(f"ph_0{i}_cd",
                                   data=np.squeeze(mri_flow.angiogram) / np.max(mri_flow.angiogram))
                grp.create_dataset(f"ph_0{i}_mag",
                                   data=np.squeeze(mri_flow.magnitude) / np.max(mri_flow.magnitude))
        del EST_TEMP, IMG_TEMP, block_count, est_temp, estimate, estimate_all, estimate_block, img, img_all, img_all1, img_block, img_temp, nopad, padded, temp, unpad

# add header from a different flow
folderf = [f'/data/data_mrcv/45_DATA_HUMANS/HEAD_NECK/TEST_RETEST_PCVIPR/DN_CODE/', ]
for folder in folderf:
    filesf = os.path.join(folder, 'Flow.h5')
    src_file = h5py.File(filesf, 'r')

foldere = [fileoutput, ]
for folder in foldere:
    filee = os.path.join(folder, 'FlowE.h5')
    dst_file_est = h5py.File(filee, 'a')

folderes = [fileoutput, ]
for folder in folderes:
    filees = os.path.join(folder, 'FlowS.h5')
    dst_file_std = h5py.File(filees, 'a')

src_group = src_file['Header']
dst_file_est.copy(src_group, dst_file_est)
dst_file_std.copy(src_group, dst_file_std)

dst_file_std.close()
dst_file_est.close()
src_file.close()

# add the averages to each dataset (I realize this is really sloppy sorry)

sum_array = np.zeros((320, 320, 320))
for i in range(nframes):
    if i < 10:
        out_name = os.path.join(folder, 'FlowS.h5')
        with h5py.File(out_name, 'a') as hf:
            dataset = hf['Data'][f'ph_00{i}_vd_1']
            sum_array += dataset
    else:
        out_name = os.path.join(folder, 'FlowS.h5')
        with h5py.File(out_name, 'a') as hf:
            dataset = hf['Data'][f'ph_0{i}_vd_1']
            sum_array += dataset
with h5py.File(out_name, 'a') as hf:
    hf['Data'].create_dataset('comp_vd_1', data=sum_array / nframes)
sum_array = np.zeros((320, 320, 320))
for i in range(nframes):
    if i < 10:
        out_name = os.path.join(folder, 'FlowS.h5')
        with h5py.File(out_name, 'a') as hf:
            dataset = hf['Data'][f'ph_00{i}_vd_2']
            sum_array += dataset
    else:
        out_name = os.path.join(folder, 'FlowS.h5')
        with h5py.File(out_name, 'a') as hf:
            dataset = hf['Data'][f'ph_0{i}_vd_2']
            sum_array += dataset
with h5py.File(out_name, 'a') as hf:
    hf['Data'].create_dataset('comp_vd_2', data=sum_array / nframes)
sum_array = np.zeros((320, 320, 320))
for i in range(nframes):
    if i < 10:
        out_name = os.path.join(folder, 'FlowS.h5')
        with h5py.File(out_name, 'a') as hf:
            dataset = hf['Data'][f'ph_00{i}_vd_3']
            sum_array += dataset
    else:
        out_name = os.path.join(folder, 'FlowS.h5')
        with h5py.File(out_name, 'a') as hf:
            dataset = hf['Data'][f'ph_0{i}_vd_3']
            sum_array += dataset
with h5py.File(out_name, 'a') as hf:
    hf['Data'].create_dataset('comp_vd_3', data=sum_array / nframes)
sum_array = np.zeros((320, 320, 320))
for i in range(nframes):
    if i < 10:
        out_name = os.path.join(folder, 'FlowS.h5')
        with h5py.File(out_name, 'a') as hf:
            dataset = hf['Data'][f'ph_00{i}_mag']
            sum_array += dataset
    else:
        out_name = os.path.join(folder, 'FlowS.h5')
        with h5py.File(out_name, 'a') as hf:
            dataset = hf['Data'][f'ph_0{i}_mag']
            sum_array += dataset
with h5py.File(out_name, 'a') as hf:
    hf['Data'].create_dataset('MAG', data=sum_array / nframes)
sum_array = np.zeros((320, 320, 320))
for i in range(nframes):
    if i < 10:
        out_name = os.path.join(folder, 'FlowS.h5')
        with h5py.File(out_name, 'a') as hf:
            dataset = hf['Data'][f'ph_00{i}_cd']
            sum_array += dataset
    else:
        out_name = os.path.join(folder, 'FlowS.h5')
        with h5py.File(out_name, 'a') as hf:
            dataset = hf['Data'][f'ph_0{i}_cd']
            sum_array += dataset
with h5py.File(out_name, 'a') as hf:
    hf['Data'].create_dataset('CD', data=sum_array / nframes)

sum_array = np.zeros((320, 320, 320))
for i in range(nframes):
    if i < 10:
        out_name = os.path.join(folder, 'FlowE.h5')
        with h5py.File(out_name, 'a') as hf:
            dataset = hf['Data'][f'ph_00{i}_vd_1']
            sum_array += dataset
    else:
        out_name = os.path.join(folder, 'FlowE.h5')
        with h5py.File(out_name, 'a') as hf:
            dataset = hf['Data'][f'ph_0{i}_vd_1']
            sum_array += dataset
with h5py.File(out_name, 'a') as hf:
    hf['Data'].create_dataset('comp_vd_1', data=sum_array / nframes)
sum_array = np.zeros((320, 320, 320))
for i in range(nframes):
    if i < 10:
        out_name = os.path.join(folder, 'FlowE.h5')
        with h5py.File(out_name, 'a') as hf:
            dataset = hf['Data'][f'ph_00{i}_vd_2']
            sum_array += dataset
    else:
        out_name = os.path.join(folder, 'FlowE.h5')
        with h5py.File(out_name, 'a') as hf:
            dataset = hf['Data'][f'ph_0{i}_vd_2']
            sum_array += dataset
with h5py.File(out_name, 'a') as hf:
    hf['Data'].create_dataset('comp_vd_2', data=sum_array / nframes)
sum_array = np.zeros((320, 320, 320))
for i in range(nframes):
    if i < 10:
        out_name = os.path.join(folder, 'FlowE.h5')
        with h5py.File(out_name, 'a') as hf:
            dataset = hf['Data'][f'ph_00{i}_vd_3']
            sum_array += dataset
    else:
        out_name = os.path.join(folder, 'FlowE.h5')
        with h5py.File(out_name, 'a') as hf:
            dataset = hf['Data'][f'ph_0{i}_vd_3']
            sum_array += dataset
with h5py.File(out_name, 'a') as hf:
    hf['Data'].create_dataset('comp_vd_3', data=sum_array / nframes)
sum_array = np.zeros((320, 320, 320))
for i in range(nframes):
    if i < 10:
        out_name = os.path.join(folder, 'FlowE.h5')
        with h5py.File(out_name, 'a') as hf:
            dataset = hf['Data'][f'ph_00{i}_mag']
            sum_array += dataset
    else:
        out_name = os.path.join(folder, 'FlowE.h5')
        with h5py.File(out_name, 'a') as hf:
            dataset = hf['Data'][f'ph_0{i}_mag']
            sum_array += dataset
with h5py.File(out_name, 'a') as hf:
    hf['Data'].create_dataset('MAG', data=sum_array / nframes)
sum_array = np.zeros((320, 320, 320))
for i in range(nframes):
    if i < 10:
        out_name = os.path.join(folder, 'FlowE.h5')
        with h5py.File(out_name, 'a') as hf:
            dataset = hf['Data'][f'ph_00{i}_cd']
            sum_array += dataset
    else:
        out_name = os.path.join(folder, 'FlowE.h5')
        with h5py.File(out_name, 'a') as hf:
            dataset = hf['Data'][f'ph_0{i}_cd']
            sum_array += dataset
with h5py.File(out_name, 'a') as hf:
    hf['Data'].create_dataset('CD', data=sum_array / nframes)

del block_crop, block_shape, block_size, block_strides, bx, by, bz, count, dataname, dataset, dst_file_est, dst_file_std, e, ef, encoding, ex, ey, ez, file, filee, filees, filesf, folder, foldere, folderes, folderf, folders, fr, fra, grp, hf, i, img_all_temp, model, model_name, mri_flow, out_name, src_file, src_group, sum_array

# %%
