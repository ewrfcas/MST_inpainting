import torch.nn as nn

from src.layers import GateConv, Conv
from src.layers import spectral_norm


def get_conv(conv_type):
    if conv_type == 'gate':
        return GateConv
    elif conv_type == 'normal':
        return Conv
    else:
        raise NotImplementedError

class Discriminator(nn.Module):
    def __init__(self, config, in_channels):
        super(Discriminator, self).__init__()

        dis_conv = get_conv(config.dis_conv_type)
        self.conv1 = nn.Sequential(
            dis_conv(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1,
                     use_spectral_norm=config.dis_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            dis_conv(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1,
                     use_spectral_norm=config.dis_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            dis_conv(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1,
                     use_spectral_norm=config.dis_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            dis_conv(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1,
                     use_spectral_norm=config.dis_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1,
                                    bias=not config.dis_spectral_norm), config.dis_spectral_norm),
        )

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        outputs = conv5

        return outputs, [conv1, conv2, conv3, conv4, conv5]


class TwinDiscriminator(nn.Module):
    def __init__(self, config, in_channels):
        super(TwinDiscriminator, self).__init__()
        self.D1 = Discriminator(config, in_channels)
        self.D2 = Discriminator(config, in_channels)

    def forward(self, x1, x2):
        x1_out, [c1, c2, c3, c4, c5] = self.D1(x1)
        x2_out, [z1, z2, z3, z4, z5] = self.D2(x2)

        return x1_out, [c1, c2, c3, c4, c5], x2_out, [z1, z2, z3, z4, z5]
