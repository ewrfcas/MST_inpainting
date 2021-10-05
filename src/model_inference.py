import os

import cv2
import numpy as np
import skimage.draw
import torchvision
import torchvision.transforms.functional
from PIL import Image
from skimage.color import gray2rgb
from skimage.color import rgb2gray
from skimage.feature import canny

from src.layers import *
from src.lsm_hawp.detector import WireframeDetector
from utils.utils import create_dir, stitch_images, to_device, to_tensor
from utils.utils import torch_show_all_params


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
        print('Decoder Params:', torch_show_all_params(generator))
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
        print('Encoder Params:', torch_show_all_params(generator))

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
        rgb1 = (rgb1 + 1) / 2  # to 0~1
        rgb2 = (rgb2 + 1) / 2
        rgb3 = (rgb3 + 1) / 2

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


def resize(img, height, width, centerCrop=False):
    imgh, imgw = img.shape[0:2]

    if centerCrop and imgh != imgw:
        # center crop
        side = np.minimum(imgh, imgw)
        j = (imgh - side) // 2
        i = (imgw - side) // 2
        img = img[j:j + side, i:i + side, ...]

    if height < imgh:
        img = cv2.resize(img, (height, width), interpolation=cv2.INTER_AREA)
    else:
        img = cv2.resize(img, (height, width))

    return img


def load_image(path, img_size, hawp_size):
    img = cv2.imread(path)[:, :, ::-1]
    origin_shape = img.shape
    origin_img = img.copy()
    if len(img.shape) < 3:
        img = gray2rgb(img)
    hawp_img = resize(img, hawp_size, hawp_size)
    img = resize(img, img_size, img_size, centerCrop=False)
    img_gray = rgb2gray(img)
    edge = canny(img_gray, sigma=2).astype(np.float)

    meta = dict()
    img = to_tensor(img)
    edge = to_tensor(edge)
    hawp_img = to_tensor(hawp_img)
    img_gray = to_tensor(img_gray)
    origin_img = to_tensor(origin_img)
    meta['img'] = img
    meta['edges'] = edge
    meta['hawp_img'] = hawp_img
    meta['img_gray'] = img_gray
    meta['origin_shape'] = origin_shape
    meta['origin_img'] = origin_img
    return meta


class MST:
    def __init__(self, config, concat_mask=False):
        self.config = config
        self.concat_mask = concat_mask
        self.lsm_hawp = WireframeDetector(is_cuda=True if str(config.DEVICE) != 'cpu' else False)
        self.lsm_hawp = self.lsm_hawp.to(config.DEVICE)
        self.img_size = config.img_size
        self.hawp_size = config.hawp_size
        self.lsm_hawp.eval()
        self.structure_encoder = SharedWEModel(config, input_channel=6)
        self.inpaint_decoder = InpaintingModel(config, input_channel=7 if concat_mask else 6)

        self.results_path = 'results'
        self.hawp_mean = torch.tensor([109.730, 103.832, 98.681]).to(config.DEVICE).reshape(1, 3, 1, 1)
        self.hawp_std = torch.tensor([22.275, 22.124, 23.229]).to(config.DEVICE).reshape(1, 3, 1, 1)

    def load(self):
        self.lsm_hawp.load_state_dict(torch.load(self.config.lsm_hawp_ckpt)['model'])
        self.structure_encoder.generator.load_state_dict(
            torch.load(self.structure_encoder.gen_weights_path)['generator'])
        self.inpaint_decoder.generator.load_state_dict(
            torch.load(self.inpaint_decoder.gen_weights_path)['generator'])

    def hawp_inference_test(self, images, masks, obj_remove=True, valid_th=0.95, mask_th=0.925):
        with torch.no_grad():
            images = images * 255.
            origin_masks = masks
            masks = F.interpolate(masks, size=(images.shape[2], images.shape[3]), mode='nearest')
            # the mask value of hawp is 127.5
            masked_images = images * (1 - masks) + torch.ones_like(images) * masks * 127.5
            images = (images - self.hawp_mean) / self.hawp_std
            masked_images = (masked_images - self.hawp_mean) / self.hawp_std

            def to_int(x):
                return tuple(map(int, x))

            lines_tensor = []
            target_mask = origin_masks.cpu().numpy()  # origin_masks, masks size is diff
            for i in range(images.shape[0]):
                lmap = np.zeros((self.img_size, self.img_size))

                output_nomask = self.lsm_hawp(images[i].unsqueeze(0))
                output_nomask = to_device(output_nomask, 'cpu')
                if output_nomask['num_proposals'] == 0:
                    lines_nomask = []
                    scores_nomask = []
                else:
                    lines_nomask = output_nomask['lines_pred'].numpy()
                    lines_nomask = [[line[1] * self.img_size, line[0] * self.img_size,
                                     line[3] * self.img_size, line[2] * self.img_size]
                                    for line in lines_nomask]
                    scores_nomask = output_nomask['lines_score'].numpy()

                output_masked = self.lsm_hawp(masked_images[i].unsqueeze(0))
                output_masked = to_device(output_masked, 'cpu')
                if output_masked['num_proposals'] == 0:
                    lines_masked = []
                    scores_masked = []
                else:
                    lines_masked = output_masked['lines_pred'].numpy()
                    lines_masked = [[line[1] * self.img_size, line[0] * self.img_size,
                                     line[3] * self.img_size, line[2] * self.img_size]
                                    for line in lines_masked]
                    scores_masked = output_masked['lines_score'].numpy()

                target_mask_ = target_mask[i, 0]
                if obj_remove:
                    for line, score in zip(lines_nomask, scores_nomask):
                        line = np.clip(line, 0, 255)
                        if score > valid_th and (
                                target_mask_[to_int(line[0:2])] == 0 or target_mask_[to_int(line[2:4])] == 0):
                            rr, cc, value = skimage.draw.line_aa(*to_int(line[0:2]), *to_int(line[2:4]))
                            lmap[rr, cc] = np.maximum(lmap[rr, cc], value)
                    for line, score in zip(lines_masked, scores_masked):
                        line = np.clip(line, 0, 255)
                        if score > mask_th and target_mask_[to_int(line[0:2])] == 1 and target_mask_[
                            to_int(line[2:4])] == 1:
                            rr, cc, value = skimage.draw.line_aa(*to_int(line[0:2]), *to_int(line[2:4]))
                            lmap[rr, cc] = np.maximum(lmap[rr, cc], value)
                else:
                    for line, score in zip(lines_masked, scores_masked):
                        if score > mask_th:
                            rr, cc, value = skimage.draw.line_aa(*to_int(line[0:2]), *to_int(line[2:4]))
                            lmap[rr, cc] = np.maximum(lmap[rr, cc], value)

                lmap = np.clip(lmap * 255, 0, 255).astype(np.uint8)
                lines_tensor.append(self.to_tensor(lmap).unsqueeze(0))

            lines_tensor = torch.cat(lines_tensor, dim=0)
        return lines_tensor.detach().to(self.config.DEVICE)

    def inference(self, path, mask_path, valid_th, mask_th, output_path=None, not_obj_remove=False):
        items = load_image(path, self.img_size, self.hawp_size)
        mask = cv2.imread(mask_path)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)[:, :, 0]
        mask = mask / 255
        items['masks'] = torchvision.transforms.functional.to_tensor(mask.astype(np.float32))
        for k in items:
            if type(items[k]) == torch.Tensor:
                items[k] = items[k].unsqueeze(0)
        items = self.cuda(items)

        self.structure_encoder.eval()
        self.inpaint_decoder.eval()

        inputs = (items['img'] * (1 - items['masks'])) + items['masks']
        hawp_lines = self.hawp_inference_test(items['hawp_img'], items['masks'],
                                              obj_remove=not not_obj_remove, valid_th=valid_th, mask_th=mask_th)
        meta_outputs = self.structure_encoder(items['img'], hawp_lines, items['edges'], items['masks'])
        edge_out = meta_outputs['edge_out'][-1]
        line_out = meta_outputs['line_out'][-1]
        lines_edges_pred = torch.clamp(line_out + edge_out, 0, 1)
        lines_preds = torch.clamp(hawp_lines + line_out, 0, 1)
        edges_pred = edge_out * items['masks'] + items['edges'] * (1 - items['masks'])
        infos = torch.cat([lines_preds, edges_pred, lines_edges_pred], dim=1)
        outputs = self.inpaint_decoder(items['img'], infos, items['masks'])
        outputs_merged = (outputs * items['masks']) + (items['img'] * (1 - items['masks']))

        # edge,line use different colors
        edges_pred = edges_pred.repeat(1, 3, 1, 1)
        x_edge = edges_pred * items['masks']
        x_edge[:, 0, :, :] = 0
        edges_pred = edges_pred * (1 - items['masks']) + x_edge

        lines_preds = lines_preds.repeat(1, 3, 1, 1)
        x_lines = lines_preds * items['masks']
        x_lines[:, 0, :, :] = 0
        lines_preds = lines_preds * (1 - items['masks']) + x_lines

        lines_edges_pred = lines_edges_pred.repeat(1, 3, 1, 1)
        x_edges_lines = lines_edges_pred * items['masks']
        x_edges_lines[:, 0, :, :] = 0
        lines_edges_pred = lines_edges_pred * (1 - items['masks']) + x_edges_lines

        images = stitch_images(
            self.postprocess(items['img']),
            [self.postprocess(inputs),
             self.postprocess(lines_preds),
             self.postprocess(edges_pred),
             self.postprocess(lines_edges_pred),
             self.postprocess(outputs_merged)],
            img_per_row=1
        )

        if output_path is None:
            name = os.path.join(self.results_path, path.split('/')[-1]).replace('.jpg', '.png')
            create_dir(self.results_path)
            print('\nsaving sample ' + name)
            images.save(name)
        else:
            images.save(output_path)

    def cuda(self, meta):
        for k in meta:
            if type(meta[k]) is torch.Tensor:
                meta[k] = meta[k].to(self.config.DEVICE)
        return meta

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        if img.shape[2] < self.img_size:
            img = F.interpolate(img, (self.img_size, self.img_size))
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()

    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = torchvision.transforms.functional.to_tensor(img).float()
        return img_t
