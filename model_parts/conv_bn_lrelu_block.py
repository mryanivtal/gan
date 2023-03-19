import torch
import torch.nn as nn


class ConvBnLreluBlock(nn.Module):
    '''
    Conv2d-BatchNorm-Relu block
    Input (named) arguments - same as ConvTranspose2d arguments:
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None
    '''

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs)-> None:
        super(ConvBnLreluBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        return self.block(input)
