import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


class GateConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, transpose=False):
        super(GateConv, self).__init__()
        self.out_channels = out_channels
        if transpose:
            self.gate_conv = nn.ConvTranspose2d(in_channels, out_channels * 2,
                                                kernel_size=kernel_size,
                                                stride=stride, padding=padding)
        else:
            self.gate_conv = nn.Conv2d(in_channels, out_channels * 2,
                                       kernel_size=kernel_size,
                                       stride=stride, padding=padding)

    def forward(self, x):
        x = self.gate_conv(x)
        (x, g) = torch.split(x, self.out_channels, dim=1)
        return x * torch.sigmoid(g)


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, transpose=False,
                 use_spectral_norm=False):
        super(Conv, self).__init__()
        self.out_channels = out_channels
        if transpose:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels,
                                           kernel_size=kernel_size, stride=stride,
                                           padding=padding, bias=not use_spectral_norm)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels,
                                  kernel_size=kernel_size, stride=stride,
                                  padding=padding, bias=not use_spectral_norm)
        if use_spectral_norm:
            self.conv = spectral_norm(self.conv)

    def forward(self, x):
        return self.conv(x)


class SNGateConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1,
                 transpose=False):
        super(SNGateConv, self).__init__()
        self.out_channels = out_channels
        if transpose:
            self.gate_conv = spectral_norm(nn.ConvTranspose2d(in_channels, out_channels * 2,
                                                              kernel_size=kernel_size, bias=False,
                                                              stride=stride, padding=padding,
                                                              dilation=dilation), mode=True)
        else:
            self.gate_conv = spectral_norm(nn.Conv2d(in_channels, out_channels * 2,
                                                     kernel_size=kernel_size, bias=False,
                                                     stride=stride, padding=padding,
                                                     dilation=dilation), mode=True)

    def forward(self, x):
        x = self.gate_conv(x)
        (x, g) = torch.split(x, self.out_channels, dim=1)
        return x * torch.sigmoid(g)


class SeparableDecoder(nn.Module):
    def __init__(self, input_channels, emb_channels, output_channel=None, stride=1):
        super(SeparableDecoder, self).__init__()

        self.emb_ch = emb_channels
        self.deconv_ch = input_channels // 2 if stride == 2 else input_channels
        self.decoder_conv = nn.Sequential(
            SNGateConv(in_channels=input_channels, out_channels=self.deconv_ch, kernel_size=3 if stride == 1 else 4,
                       stride=stride, padding=1, transpose=True if stride > 1 else False),
            nn.InstanceNorm2d(self.deconv_ch, track_running_stats=False),
            nn.ReLU(True)
        )
        self.emb_head = nn.Sequential(
            nn.Conv2d(self.deconv_ch, emb_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(emb_channels * 2, track_running_stats=False),
            nn.ReLU(True)
        )
        self.att_head = nn.Sequential(
            nn.Conv2d(emb_channels * 2, emb_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(emb_channels, track_running_stats=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=emb_channels, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.to_edge = nn.Sequential(
            nn.Conv2d(emb_channels, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.to_line = nn.Sequential(
            nn.Conv2d(emb_channels, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.to_rgb = nn.Sequential(
            nn.Conv2d(emb_channels, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
        if output_channel is not None:
            self.proj = nn.Conv2d(in_channels=self.deconv_ch + emb_channels,
                                  out_channels=output_channel, kernel_size=1, stride=1, padding=0)
        else:
            self.proj = None

    def forward(self, x):
        x = self.decoder_conv(x)
        emb = self.emb_head(x)
        e, l = torch.split(emb, self.emb_ch, dim=1)
        edge = self.to_edge(e)
        line = self.to_line(l)
        att = self.att_head(emb)
        x_combine = e * att + l * (1 - att)
        rgb = self.to_rgb(x_combine)
        # rgb = (rgb + 1) / 2
        if self.proj:
            x_out = torch.cat([x, x_combine], dim=1)  # deconv_ch+emb
            x_out = self.proj(x_out)
            return x_out, rgb, edge, line, att
        else:
            return rgb, edge, line, att


class EfficientAttention(nn.Module):
    def __init__(self, in_channels, dim, head_count, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.head_count = head_count
        self.dim = dim

        self.keys = nn.Conv2d(in_channels, dim, 1)
        self.queries = nn.Conv2d(in_channels, dim, 1)
        self.values = nn.Conv2d(in_channels, dim, 1)
        if dim != out_channels:
            self.reprojection = nn.Conv2d(dim, out_channels, 1)
        else:
            self.reprojection = None

    def forward(self, input_, mask=None, return_scores=False):
        n, _, h, w = input_.size()
        keys = self.keys(input_)
        queries = self.queries(input_)
        values = self.values(input_)
        head_channels = self.dim // self.head_count

        if mask is not None:
            # [b,1,h,w]
            mask = F.interpolate(mask, size=[h, w], mode='nearest')
            keys += (mask * -10000.0)
            queries += (mask * -10000.0)

        keys = keys.reshape((n, self.dim, h * w))  # [b,d,h*w]
        queries = queries.reshape(n, self.dim, h * w)
        values = values.reshape((n, self.dim, h * w))

        attended_values = []
        scores = 0
        for i in range(self.head_count):
            key = F.softmax(keys[:, i * head_channels: (i + 1) * head_channels, :], dim=2)
            query = F.softmax(queries[:, i * head_channels: (i + 1) * head_channels, :], dim=1)
            value = values[:, i * head_channels: (i + 1) * head_channels, :]
            context = key @ value.transpose(1, 2)  # [b, d, d]
            attended_value = (context.transpose(1, 2) @ query).reshape(n, head_channels, h, w)
            attended_values.append(attended_value)
            if return_scores:
                score = torch.matmul(query.transpose(1, 2), key)  # [b, hw, hw]
                score = torch.mean(score, dim=1).reshape([n, h, w])
                scores += score

        aggregated_values = torch.cat(attended_values, dim=1)
        if self.reprojection is not None:
            reprojected_value = self.reprojection(aggregated_values)
        else:
            reprojected_value = aggregated_values

        attention = reprojected_value + input_

        if return_scores:
            max_value, _ = torch.max(scores.reshape([n, h * w]), dim=1)
            max_value = max_value[:, None, None]
            scores = scores / (max_value + 1e-5)
            scores = scores.unsqueeze(1)
            scores = scores.detach()
            return attention, scores
        else:
            return attention


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module
