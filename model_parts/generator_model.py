import torch
import torch.nn as nn
from model_parts.convt_bn_relu_block import ConvtBnReluBlock


class Generator(nn.Module):
    '''
    DC-GAN generator module
    '''
    def __init__(self, num_channels, num_feature_maps, latent_dim)-> None:
        super(Generator, self).__init__()

        self.generator = nn.Sequential(
            ConvtBnReluBlock(latent_dim, num_feature_maps * 8, 4, stride=1, padding=0, bias=False),
            ConvtBnReluBlock(num_feature_maps * 8, num_feature_maps * 4, 4, stride=2, padding=1, bias=False),
            ConvtBnReluBlock(num_feature_maps * 4, num_feature_maps * 2, 4, stride=2, padding=1, bias=False),
            ConvtBnReluBlock(num_feature_maps * 2, num_feature_maps, 4, stride=2, padding=1, bias=False),

            nn.ConvTranspose2d(num_feature_maps, num_channels, 4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.generator(input)
