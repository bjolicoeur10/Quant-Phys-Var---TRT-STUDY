import torch

def extract_patches_3ds(x, kernel_size, padding=(0, 0, 0), stride=(1, 1, 1)):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    if len(padding) == 3:
        padding = (padding[0], padding[0], padding[1], padding[1], padding[2], padding[2])
    if isinstance(stride, int):
        stride = (stride, stride, stride)

    x = torch.nn.functional.pad(x, padding)
    channels = x.shape[1]

    x = x.unfold(2, kernel_size[0], stride[0]).unfold(3, kernel_size[1], stride[1]).unfold(4, kernel_size[2], stride[2])
    # (B, C, d_dim_out, h_dim_out, w_dim_out, kernel_size[0], kernel_size[1], kernel_size[2])
    x = x.contiguous().view(-1, channels, kernel_size[0], kernel_size[1], kernel_size[2])
    # (B * d_dim_out * h_dim_out * w_dim_out, C, kernel_size[0], kernel_size[1], kernel_size[2])
    return x


def extract_patches_3d(x, kernel_size, padding=(0, 0, 0), stride=(1, 1, 1)):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    if len(padding) == 3:
        padding = (padding[0], padding[0], padding[1], padding[1], padding[2], padding[2])
    if isinstance(stride, int):
        stride = (stride, stride, stride)

    def get_dim_blocks(dim_in, dim_kernel, dim_padding=0, dim_stride=1, dim_dilation=1):
        dim_out = (dim_in + 2 * dim_padding - dim_dilation * (dim_kernel - 1) - 1) // dim_stride + 1
        return dim_out

    x = torch.nn.functional.pad(x, padding)
    channels = x.shape[1]

    d_dim_in = x.shape[2]
    h_dim_in = x.shape[3]
    w_dim_in = x.shape[4]
    d_dim_out = get_dim_blocks(d_dim_in, kernel_size[0], dim_stride=stride[0])
    h_dim_out = get_dim_blocks(h_dim_in, kernel_size[1], dim_stride=stride[1])
    w_dim_out = get_dim_blocks(w_dim_in, kernel_size[2], dim_stride=stride[2])
    # print(d_dim_in, h_dim_in, w_dim_in, d_dim_out, h_dim_out, w_dim_out)

    # (B, C, D, H, W)
    x = x.view(-1, channels, d_dim_in, h_dim_in * w_dim_in)
    # (B, C, D, H * W)

    x = torch.nn.functional.unfold(x, kernel_size=(kernel_size[0], 1), stride=(stride[0], 1))
    # (B, C * kernel_size[0], d_dim_out * H * W)

    x = x.view(-1, channels * kernel_size[0] * d_dim_out, h_dim_in, w_dim_in)
    # (B, C * kernel_size[0] * d_dim_out, H, W)

    x = torch.nn.functional.unfold(x, kernel_size=(kernel_size[1], kernel_size[2]), stride=(stride[1], stride[2]))
    # (B, C * kernel_size[0] * d_dim_out * kernel_size[1] * kernel_size[2], h_dim_out * w_dim_out)

    x = x.view(-1, channels, kernel_size[0], d_dim_out, kernel_size[1], kernel_size[2], h_dim_out, w_dim_out)
    # (B, C, kernel_size[0], d_dim_out, kernel_size[1], kernel_size[2], h_dim_out, w_dim_out)

    x = x.permute(0, 1, 3, 6, 7, 2, 4, 5)
    # (B, C, d_dim_out, h_dim_out, w_dim_out, kernel_size[0], kernel_size[1], kernel_size[2])

    x = x.contiguous().view(-1, channels, kernel_size[0], kernel_size[1], kernel_size[2])
    # (B * d_dim_out * h_dim_out * w_dim_out, C, kernel_size[0], kernel_size[1], kernel_size[2])

    return x


def combine_patches_3d(x, kernel_size, output_shape, padding=(0, 0, 0), stride=(1, 1, 1)):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    if len(padding) == 3:
        padding = (padding[0], padding[0], padding[1], padding[1], padding[2], padding[2])
    if isinstance(stride, int):
        stride = (stride, stride, stride)

    def get_dim_blocks(dim_in, dim_kernel, dim_padding=0, dim_stride=1, dim_dilation=1):
        dim_out = (dim_in + 2 * dim_padding - dim_dilation * (dim_kernel - 1) - 1) // dim_stride + 1
        return dim_out

    x = torch.nn.functional.pad(x, padding)

    channels = x.shape[1]
    d_dim_out, h_dim_out, w_dim_out = output_shape[2:]
    d_dim_in = get_dim_blocks(d_dim_out, kernel_size[0], dim_stride=stride[0])
    h_dim_in = get_dim_blocks(h_dim_out, kernel_size[1], dim_stride=stride[1])
    w_dim_in = get_dim_blocks(w_dim_out, kernel_size[2], dim_stride=stride[2])
    # print(d_dim_in, h_dim_in, w_dim_in, d_dim_out, h_dim_out, w_dim_out)

    x = x.view(-1, channels, d_dim_in, h_dim_in, w_dim_in, kernel_size[0], kernel_size[1], kernel_size[2])
    # (B, C, d_dim_in, h_dim_in, w_dim_in, kernel_size[0], kernel_size[1], kernel_size[2])

    x = x.permute(0, 1, 5, 2, 6, 7, 3, 4)
    # (B, C, kernel_size[0], d_dim_in, kernel_size[1], kernel_size[2], h_dim_in, w_dim_in)

    x = x.contiguous().view(-1, channels * kernel_size[0] * d_dim_in * kernel_size[1] * kernel_size[2],
                            h_dim_in * w_dim_in)
    # (B, C * kernel_size[0] * d_dim_in * kernel_size[1] * kernel_size[2], h_dim_in * w_dim_in)

    x = torch.nn.functional.fold(x, output_size=(h_dim_out, w_dim_out), kernel_size=(kernel_size[1], kernel_size[2]),
                                 stride=(stride[1], stride[2]))
    # (B, C * kernel_size[0] * d_dim_in, H, W)

    x = x.view(-1, channels * kernel_size[0], d_dim_in * h_dim_out * w_dim_out)
    # (B, C * kernel_size[0], d_dim_in * H * W)

    x = torch.nn.functional.fold(x, output_size=(d_dim_out, h_dim_out * w_dim_out), kernel_size=(kernel_size[0], 1),
                                 stride=(stride[0], 1))
    # (B, C, D, H * W)

    x = x.view(-1, channels, d_dim_out, h_dim_out, w_dim_out)
    # (B, C, D, H, W)

    return x
