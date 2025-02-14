import torch
import torch.nn.functional as F
from math import exp


__all__ = [ 'ssim', 'SSIM', 'SSIM_loss', 'angio_weighted_loss', 'complex_mse']

def complex_mse(estimate, truth):
    return torch.mean(torch.abs(truth - estimate)**2)

def angio_weighted_loss(estimate, truth):
    # Input is batchsize, 8, Nx, Ny

    # Get the angiogram
    # real_truth, imag_truth = utils.channels2realimag(truth,-3)
    # diff = (real_truth - real_truth[:,0].unsqueeze(1))**2
    # diff+= (imag_truth - imag_truth[:,0].unsqueeze(1))**2

    diff = truth - truth[:,0].unsqueeze(1)
    angio = torch.sum(torch.abs(diff)**2, dim=1, keepdim=True) ** 0.5
    angio /= torch.mean(angio)

    # Calculate phase difference
    diff = torch.mean(torch.abs(angio*(truth - estimate))**2 )

    return diff

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=False, val_range=1):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window.to(img1.device).type(img1.dtype)
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        ssimv = ssim(img1, img2, window=window, window_size=self.window_size,
                    size_average=self.size_average, val_range=self.val_range)
        ssimv = torch.reshape( ssimv, (-1, 1))

        return ssimv


class SSIM_loss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=False, val_range=1):
        super(SSIM_loss, self).__init__()

        # SSIM operator (batchwise)
        self.ssim_op = SSIM(window_size, size_average, val_range)

    def forward(self, img1, img2):
        batch_ssim = self.ssim_op(img1, img2)
        loss = torch.mean(1.0 - batch_ssim)
        return loss