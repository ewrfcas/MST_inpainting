import os

from src.layers import *
from utils import torch_show_all_params


class BaseModel(nn.Module):
    def __init__(self, name, config):
        super(BaseModel, self).__init__()

        self.config = config
        self.iteration = 0
        self.name = name
        self.gen_weights_path = os.path.join(config.PATH, name + '_gen.pth')


class InpaintingModel(BaseModel):
    def __init__(self, config, input_channel=4):
        super(InpaintingModel, self).__init__('InpaintingModel', config)
        self.input_channel = input_channel
        generator = InpaintGateGenerator(input_channel=input_channel,
                                         inpaint_attention=False)
        print('Decoder:')
        torch_show_all_params(generator)
        generator = generator.to(config.DEVICE)

        self.add_module('generator', generator)

    def forward(self, images, infos, masks):
        images_masked = (images * (1 - masks).float()) + masks
        if self.input_channel == 7:
            inputs = torch.cat([images_masked, infos, masks], dim=1)
        else:
            inputs = torch.cat((images_masked, infos), dim=1)
        outputs = self.generator(inputs)
        return outputs


class SharedWEModel(BaseModel):
    def __init__(self, config, input_channel=6):
        super(SharedWEModel, self).__init__('SharedWEModel', config)

        self.model = config.MODEL
        generator = SWEGenerator(input_channel=input_channel)
        print('Encoder:')
        torch_show_all_params(generator)

        generator = generator.to(config.DEVICE)

        self.add_module('generator', generator)

    def forward(self, images, lines, edges, masks):
        meta_outputs = {}
        images_masked = (images * (1 - masks).float()) + masks
        edges_masked = (edges * (1 - masks).float()) + masks
        inputs = torch.cat([images_masked, lines, edges_masked, masks], dim=1)
        [rgb1, rgb2, rgb3], \
        [edge1, edge2, edge3], \
        [line1, line2, line3], att = self.generator(inputs, masks)
        meta_outputs['img_out'] = [rgb1, rgb2, rgb3]
        meta_outputs['edge_out'] = [edge1, edge2, edge3]
        meta_outputs['line_out'] = [line1, line2, line3]
        meta_outputs['att_score'] = att

        return meta_outputs

class InpaintGateGenerator(BaseNetwork):
    def __init__(self, residual_blocks=8, init_weights=True, input_channel=4, output_channel=3,
                 inpaint_attention=False):
        super(InpaintGateGenerator, self).__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            GateConv(in_channels=input_channel, out_channels=64, kernel_size=7, padding=0),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            GateConv(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            GateConv(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True)
        )

        self.inpaint_attention = inpaint_attention
        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2)
            blocks.append(block)
        self.middle = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            GateConv(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, transpose=True),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            GateConv(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, transpose=True),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=output_channel, kernel_size=7, padding=0),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x, mask=None):
        x = self.encoder(x)
        if self.inpaint_attention:
            x = self.middle1(x)
            x, _ = self.att_mid(x, mask)
            x = self.middle2(x)
        else:
            x = self.middle(x)
        x = self.decoder(x)
        x = (torch.tanh(x) + 1) / 2

        return x


class SWEGenerator(BaseNetwork):
    def __init__(self, input_channel=6, init_weights=True):
        super(SWEGenerator, self).__init__()
        ch = 64
        self.ch = ch
        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            SNGateConv(in_channels=input_channel, out_channels=ch, kernel_size=7, padding=0),
            nn.InstanceNorm2d(ch, track_running_stats=False),
            nn.ReLU(True),

            SNGateConv(in_channels=ch, out_channels=ch * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ch * 2, track_running_stats=False),
            nn.ReLU(True),

            SNGateConv(in_channels=ch * 2, out_channels=ch * 4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ch * 4, track_running_stats=False),
            nn.ReLU(True)
        )

        blocks = []
        for i in range(4):
            block = ResnetBlock(dim=ch * 4, dilation=2)
            blocks.append(block)
        self.middle1 = nn.Sequential(*blocks)
        self.attention = EfficientAttention(ch * 4, ch * 4, head_count=4, out_channels=ch * 4)
        blocks = []
        for i in range(4):
            block = ResnetBlock(dim=ch * 4, dilation=2)
            blocks.append(block)
        self.middle2 = nn.Sequential(*blocks)

        self.separable_decoder1 = SeparableDecoder(ch * 4, ch * 2, output_channel=ch * 3, stride=1)
        self.separable_decoder2 = SeparableDecoder(ch * 3, ch, output_channel=ch * 2, stride=2)
        self.separable_decoder3 = SeparableDecoder(ch * 2, ch, output_channel=None, stride=2)

        if init_weights:
            self.init_weights()

    def forward(self, x, mask=None):
        x = self.encoder(x)
        x = self.middle1(x)
        x = self.attention(x, mask, return_scores=False)
        x = self.middle2(x)

        x, rgb1, edge1, line1, a1 = self.separable_decoder1(x)
        x, rgb2, edge2, line2, a2 = self.separable_decoder2(x)
        rgb3, edge3, line3, a3 = self.separable_decoder3(x)

        return [rgb1, rgb2, rgb3], [edge1, edge2, edge3], [line1, line2, line3], [a1, a2, a3]


class Discriminator(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]
