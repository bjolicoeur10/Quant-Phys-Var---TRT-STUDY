import numpy as np
import torch

__all__ = ['complex2channels2D', 'channels2realimag']

def complex2channels2D(input, dim=0):
    # input is cuda array, complex64
    # output is cuda array, float32, (,2)

    if torch.is_tensor(input):
        output = torch.stack([torch.real(input), torch.imag(input)], dim=dim)
        return output

    output = np.stack([np.real(input), np.imag(input)], axis=dim)

    return output

def channels2realimag(input,dim=0):
    # Input is channels stacked real real real imag imag imag (e.g. for 3 complex images)
    if torch.is_tensor(input):
        real, imag = torch.split(input, input.shape[dim]//2, dim=dim)
    else:
        real, imag = np.split(input, input.shape[dim]//2, axis=dim)

    return real, imag

