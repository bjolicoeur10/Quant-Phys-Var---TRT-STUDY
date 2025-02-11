import torch
import os
import h5py as h5
import numpy as np

train_file_folder = 'Q:/DeepFlowData2'
log_dir = os.path.join('E:/Deep_Flow/Log')

def load_case(name):
    # Test load the data
    with h5.File(name, 'r') as hf:

        imgs = []
        for c in list(hf['Images'].keys()):
            #print(f'Read {c}')
            try:
                imR = np.array(hf['Images'][c]['real'])
                imI = np.array(hf['Images'][c]['imag'])
                imgs.append(imR + 1j * imI)
            except:
                imC = np.array(hf['Images'][c])
                print(imC.dtype)
                imgs.append(imC)

        imgs = np.stack(imgs)

    return imgs


def load_case_2ch(name):

    # Test load the data
    with h5.File(name, 'r') as hf:
        imgs = []
        for c in list(hf['Images'].keys()):
            print(f'Read {c}')
            imR = np.array(hf['Images'][c]['real'])
            imI = np.array(hf['Images'][c]['imag'])
            im2ch = np.stack([imR, imI], axis=0)
            imgs.append(im2ch)
        imgs = np.stack(imgs, axis=0)

    return imgs

