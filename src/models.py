import os

from src.discriminators import TwinDiscriminator, Discriminator
from src.layers import *
from src.loss import AdversarialLoss, VGG19, PerceptualLoss, StyleLoss


class InpaintingModel(nn.Module):
    def __init__(self, config, input_channel=7):
        super(InpaintingModel, self).__init__()
        self.iteration = 0
        self.g_path = os.path.join(config.path, 'DEC_G')
        self.d_path = os.path.join(config.path, 'DEC_D')
        self.config = config
        self.g_model = InpaintGateGenerator(input_channel=input_channel,
                                            inpaint_attention=False).to(config.device)
        self.d_model = Discriminator(config, in_channels=3)
        self.l1_loss = nn.L1Loss('none').to(config.device)
        vgg = VGG19(pretrained=True, vgg_norm=True).to(config.device)
        vgg.eval()
        self.perceptual_loss = PerceptualLoss(vgg, weights=config.vgg_loss_weight, reduction='none').to(config.device)
        self.style_loss = StyleLoss(vgg).to(config.device)
        self.adversarial_loss = AdversarialLoss(type=config.gan_type).to(config.device)

    def forward(self, images, infos, masks):
        images_masked = (images * (1 - masks).float()) + masks
        inputs = torch.cat([images_masked, infos, masks], dim=1)
        outputs = self.g_model(inputs)
        return outputs

    def process(self, images, edges, lines, masks):
        self.iteration += 1

        gen_loss = 0
        dis_loss = 0

        edge_line_maps = torch.clamp(edges + lines, 0, 1.0)
        infos = torch.cat([edges, lines, edge_line_maps], dim=1)
        outputs = self.forward(images, infos, masks)

        # discriminator loss
        dis_input_real = images
        dis_input_fake = outputs.detach()
        dis_real, _ = self.d_model(dis_input_real)
        dis_fake, _ = self.d_model(dis_input_fake)
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2

        # generator adversarial loss
        gen_input_fake = outputs
        gen_fake, _ = self.d_model(gen_input_fake)  # in: [rgb(3)]
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.config.adv_loss_weight
        gen_loss += gen_gan_loss

        # generator l1 loss
        gen_l1_loss = self.l1_loss(outputs, images)  # [bs, 3, H, W]
        masks_ = masks.repeat(1, 3, 1, 1)
        mask_sum = torch.sum(masks_, dim=[2, 3]) + 1e-7
        sub_mask_sum = torch.sum(1 - masks_, dim=[2, 3]) + 1e-7
        m_gen_l1_loss = torch.mean(torch.sum(gen_l1_loss * masks_, dim=[2, 3]) / mask_sum)
        nom_gen_l1_loss = torch.mean(torch.sum(gen_l1_loss * (1 - masks_), dim=[2, 3]) / sub_mask_sum)
        gen_l1_loss = (m_gen_l1_loss + nom_gen_l1_loss) * 0.5 * self.config.l1_loss_weight
        gen_loss += gen_l1_loss

        # generator perceptual loss
        gen_content_loss = self.perceptual_loss(outputs, images)
        perceptual_loss = 0.0
        for content_loss in gen_content_loss:
            masks_ = F.interpolate(masks, size=(content_loss.shape[2], content_loss.shape[3]))
            masks_ = masks_.repeat(1, content_loss.shape[1], 1, 1)
            mask_sum = torch.sum(masks_, dim=[2, 3]) + 1e-7
            sub_mask_sum = torch.sum(1 - masks_, dim=[2, 3]) + 1e-7
            m_content_loss = torch.mean(torch.sum(content_loss * masks_, dim=[2, 3]) / mask_sum)
            nom_content_loss = torch.mean(torch.sum(content_loss * (1 - masks_), dim=[2, 3]) / sub_mask_sum)
            perceptual_loss += (m_content_loss + nom_content_loss) * 0.5
        gen_content_loss = perceptual_loss

        gen_content_loss = gen_content_loss * self.config.content_loss_weight
        gen_loss += gen_content_loss

        # generator style loss
        gen_style_loss = self.style_loss(outputs * masks, images * masks)
        gen_style_loss = gen_style_loss * self.config.style_loss_weight
        gen_loss += gen_style_loss

        # create logs
        logs = [("l_d", dis_loss.item()),
                ("l_g", gen_gan_loss.item()),
                ("l_l1", gen_l1_loss.item()),
                ("l_per", gen_content_loss.item()),
                ("l_sty", gen_style_loss.item())]

        return outputs, gen_loss, dis_loss, logs


class SharedWEModel(nn.Module):
    def __init__(self, config, input_channel=6, image_output_channel=3):
        super(SharedWEModel, self).__init__()
        self.g_path = os.path.join(config.path, 'PSS_G')
        self.d_path = os.path.join(config.path, 'PSS_D')
        self.config = config
        self.iteration = 0

        self.g_model = SWEGenerator(input_channel=input_channel).to(config.device)
        self.d_model = TwinDiscriminator(config, in_channels=image_output_channel).to(config.device)
        self.l1_loss = nn.L1Loss('mean').to(config.device)
        self.adversarial_loss = AdversarialLoss(type=config.gan_type).to(config.device)

    def forward(self, images, lines, edges, masks):
        meta_outputs = {}
        images_masked = (images * (1 - masks).float()) + masks
        edges_masked = (edges * (1 - masks).float())
        inputs = torch.cat([images_masked, lines, edges_masked, masks], dim=1)
        [rgb1, rgb2, rgb3], [edge1, edge2, edge3], [line1, line2, line3], att = self.g_model(inputs, masks)
        meta_outputs['img_out'] = [rgb1, rgb2, rgb3]
        meta_outputs['edge_out'] = [edge1, edge2, edge3]
        meta_outputs['line_out'] = [line1, line2, line3]
        meta_outputs['att_score'] = att
        meta_outputs['images_masked'] = images_masked
        meta_outputs['edges_masked'] = edges_masked
        meta_outputs['lines_masked'] = lines

        return meta_outputs

    def process(self, images, lines, edges, masks, real_lines):
        # images: input images (rgb)
        # lcnn_lines: lines predicted from lcnn
        # edges: groundtruth edges from canny
        # lines: groundtruth wireframe
        # lines_edges: edges+lines
        # masks: mask area
        self.iteration += 1
        # process outputs
        meta_outputs = self.forward(images, lines, edges, masks)

        img_out = meta_outputs['img_out']  # [B,3,H,W] not opt by gan
        edge_out = meta_outputs['edge_out']  # [B,3,H,W]
        line_out = meta_outputs['line_out']

        # divide edge and line
        gen_loss = 0
        dis_loss = 0

        gt_edge = [edges]
        edges_temp = F.conv2d(edges, torch.ones((1, 1, 2, 2), dtype=edges.dtype, device=edges.device) * 0.5)
        for i in range(2, 0, -1):
            edges_temp_ = F.interpolate(edges_temp, size=[64 * i, 64 * i], mode='bilinear', align_corners=False)
            edges_temp_ = F.interpolate(edges_temp_, size=[256, 256])
            edges_temp_ = torch.clamp(edges_temp_, 0, 1)
            gt_edge.append(edges_temp_)
        gt_edge = torch.cat(gt_edge, dim=1)
        pred_edge = torch.cat([edge_out[-1], F.interpolate(edge_out[-2], size=[256, 256]),
                               F.interpolate(edge_out[-3], size=[256, 256])], dim=1)

        gt_line = [real_lines]
        lines_temp = F.conv2d(real_lines, torch.ones((1, 1, 2, 2), dtype=real_lines.dtype,
                                                     device=real_lines.device) * 0.5)
        for i in range(2, 0, -1):
            lines_temp_ = F.interpolate(lines_temp, size=[64 * i, 64 * i], mode='bilinear', align_corners=False)
            lines_temp_ = F.interpolate(lines_temp_, size=[256, 256])
            lines_temp_ = torch.clamp(lines_temp_, 0, 1)
            gt_line.append(lines_temp_)
        gt_line = torch.cat(gt_line, dim=1)
        pred_line = torch.cat([line_out[-1], F.interpolate(line_out[-2], size=[256, 256]),
                               F.interpolate(line_out[-3], size=[256, 256])], dim=1)

        # discriminator loss
        edge_real, edge_real_feat, line_real, line_real_feat = self.d_model(gt_edge, gt_line)
        edge_fake, edge_fake_feat, line_fake, line_fake_feat = self.d_model(pred_edge.detach(),
                                                                            pred_line.detach())
        D_edge_real_loss = self.adversarial_loss(edge_real, True, True)
        D_edge_fake_loss = self.adversarial_loss(edge_fake, False, True)
        D_line_real_loss = self.adversarial_loss(line_real, True, True)
        D_line_fake_loss = self.adversarial_loss(line_fake, False, True)
        dis_loss += (D_edge_real_loss + D_edge_fake_loss + D_line_real_loss + D_line_fake_loss) / 4

        # generator adversarial loss
        G_edge_fake, G_edge_fake_feat, G_line_fake, G_line_fake_feat = self.d_model(pred_edge, pred_line)
        G_edge_gan_loss = self.adversarial_loss(G_edge_fake, True, False) * self.config.adv_loss_weight
        G_line_gan_loss = self.adversarial_loss(G_line_fake, True, False) * self.config.adv_loss_weight
        gen_gan_loss = (G_edge_gan_loss + G_line_gan_loss) / 2
        gen_loss += gen_gan_loss

        # generator feature matching loss
        gen_fm_loss = 0
        # edge fm
        for i in range(len(edge_real_feat)):
            gen_fm_loss += self.l1_loss(G_edge_fake_feat[i], edge_real_feat[i].detach())

        for i in range(len(line_real_feat)):
            gen_fm_loss += self.l1_loss(G_line_fake_feat[i], line_real_feat[i].detach())
        gen_fm_loss = gen_fm_loss * self.config.fm_loss_weight

        gen_loss += gen_fm_loss

        img_l1_loss = self.l1_loss(img_out[-1], images) * self.config.l1_loss_weight / torch.mean(masks)
        gen_loss += img_l1_loss

        # pyramid rgb loss without the last layer
        pyramid_rgb_loss = 0
        for pred_rgb in img_out[:-1]:
            pyramid_rgb_loss += self.l1_loss(pred_rgb, F.interpolate(images, size=pred_rgb.size()[2:4],
                                                                     mode='bilinear', align_corners=True)).mean()
        pyramid_rgb_loss = pyramid_rgb_loss * self.config.pyramid_loss_weight
        gen_loss += pyramid_rgb_loss

        # create logs
        logs = [
            ("l_d", dis_loss.item()),
            ("l_g", gen_gan_loss.item()),
            ("l_img_l1", img_l1_loss.item()),
            ('l_py_rgb', pyramid_rgb_loss.item()),
        ]

        if gen_fm_loss != 0:
            logs.append(('l_fm', gen_fm_loss.item()))

        return gen_loss, dis_loss, logs


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
        x = torch.tanh(x)

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
