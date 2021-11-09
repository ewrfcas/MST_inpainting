import glob
import os
import pickle
import random

import cv2
import numpy as np
import skimage.draw
import torch
import torchvision.transforms.functional as F
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import canny
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate


class LSMDataset(torch.utils.data.Dataset):
    def __init__(self, config, flist, wireframe_path=None, irr_mask_path=None, seg_mask_path=None,
                 fix_mask_path=None, wireframe_mask_rate=0.5, hawp_th=0.95, training=True):
        super(LSMDataset, self).__init__()
        self.config = config
        self.training = training
        self.data = self.load_flist(flist)
        self.wireframe_path = wireframe_path
        self.wireframe_mask_rate = wireframe_mask_rate
        self.hawp_th = hawp_th
        self.irr_masks = []
        self.seg_masks = []
        self.fix_masks = []
        self.hawp_height = 512
        self.hawp_width = 512
        if irr_mask_path is not None:
            self.irr_masks = self.load_flist(irr_mask_path)
        if seg_mask_path is not None:
            self.seg_masks = self.load_flist(seg_mask_path)
        if fix_mask_path is not None:
            self.fix_masks = self.load_flist(fix_mask_path)
        if wireframe_path is not None:
            self.wireframe_results = pickle.load(open(wireframe_path, 'rb'))
        else:
            self.wireframe_results = None
        if not training:
            assert len(self.fix_masks) == len(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.load_item(index)

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    def load_edge(self, img):

        return canny(img, sigma=2, mask=None).astype(np.float)

    def load_item(self, index):

        size = self.config.input_size
        fname = os.path.basename(self.data[index]).split('.')[0]
        # load image
        img = cv2.imread(self.data[index])
        while img is None:
            print('Bad image {}...'.format(self.data[index]))
            index = random.randint(0, len(self.data) - 1)
            img = cv2.imread(self.data[index])
        img = img[:, :, ::-1]

        # resize/crop if needed
        img = self.resize(img, size, size, center_crop=self.config.center_crop)
        # load mask
        mask = self.load_mask(img, index)
        # augment mask
        if self.training is True and self.config.flip is True:
            if random.random() < 0.5:
                mask = mask[:, ::-1, ...]
            if random.random() < 0.5:
                mask = mask[::-1, :, ...]

        if self.wireframe_results is not None:  # load train wireframe results
            real_line = self.load_all_wireframe(fname, size, th=self.hawp_th)
            line = self.load_masked_wireframe(fname, size, mask, th=self.hawp_th,
                                              mask_rate=self.wireframe_mask_rate)
        else:  # for inference, use img with 512x512 as inputs for wireframe detector
            real_line = None
            line = cv2.resize(img, (512, 512))

        # augment data
        if self.training is True and self.config.flip is True:
            if random.random() < 0.5:
                img = img[:, ::-1, ...]
                line = line[:, ::-1, ...]
                if real_line is not None:
                    real_line = real_line[:, ::-1, ...]

        img_gray = rgb2gray(img)
        edge = self.load_edge(img_gray)

        img = self.to_tensor(img, norm=True)  # norm to -1~1
        edge = self.to_tensor(edge)
        line = self.to_tensor(line)  # always 0~1 for 512x512 img (test) or 256x256 lines (train)
        if real_line is not None:
            real_line = self.to_tensor(real_line)
        mask = self.to_tensor(mask)
        meta = {'img': img, 'mask': mask, 'edge': edge, 'line': line, 'real_line': real_line,
                'name': os.path.basename(self.data[index])}

        return meta

    def load_masked_wireframe(self, fname, input_size, target_mask, th=0.95, mask_rate=0.5):

        def to_int(x):
            return tuple(map(int, x))

        result = self.wireframe_results[fname]
        lines_pred = np.clip(result['lines_pred'], 0, 1)
        lines_score = result['lines_score']
        lines_pred = [[line[1] * input_size, line[0] * input_size,
                       line[3] * input_size, line[2] * input_size] for line in lines_pred]
        lmap = np.zeros((input_size, input_size), dtype=np.float32)
        for line, score in zip(lines_pred, lines_score):
            masked_points = 0
            p1 = np.clip(to_int(line[0:2]), 0, input_size - 1)
            p2 = np.clip(to_int(line[2:4]), 0, input_size - 1)
            if target_mask[p1[0], p1[1]] != 0:
                masked_points += 1
            if target_mask[p2[0], p2[1]] != 0:
                masked_points += 1
            if score > th and masked_points <= 1:
                # if 0 endpoint is masked, maintain the line, if 1 endpoint is masked, maintain it with mask_rate
                if masked_points == 0 or random.random() > mask_rate:
                    rr, cc, value = skimage.draw.line_aa(*to_int(line[0:2]), *to_int(line[2:4]))
                    rr = np.clip(rr, 0, input_size - 1)
                    cc = np.clip(cc, 0, input_size - 1)
                    lmap[rr, cc] = np.maximum(lmap[rr, cc], value)

        return lmap

    def load_all_wireframe(self, fname, input_size, th=0.95):

        def to_int(x):
            return tuple(map(int, x))

        result = self.wireframe_results[fname]
        lines_pred = np.clip(result['lines_pred'], 0, 1)
        lines_score = result['lines_score']
        lines_pred = [[line[1] * input_size, line[0] * input_size,
                       line[3] * input_size, line[2] * input_size] for line in lines_pred]
        lmap = np.zeros((input_size, input_size), dtype=np.float32)
        for line, score in zip(lines_pred, lines_score):
            if score > th:
                rr, cc, value = skimage.draw.line_aa(*to_int(line[0:2]), *to_int(line[2:4]))
                rr = np.clip(rr, 0, input_size - 1)
                cc = np.clip(cc, 0, input_size - 1)
                lmap[rr, cc] = np.maximum(lmap[rr, cc], value)

        return lmap

    def load_mask(self, img, index):
        imgh, imgw = img.shape[0:2]

        # test mode: load mask non random
        if self.training is False:
            mask = cv2.imread(self.fix_masks[index], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (imgw, imgh),
                              interpolation=cv2.INTER_NEAREST)
            mask = (mask > 127).astype(np.uint8) * 255
            return mask
        else:  # train mode: 40% mask with random brush, 40% mask with coco mask, 20% with additions
            rdv = random.random()
            if rdv < self.config.mask_rates[0]:
                mask_index = random.randint(0, len(self.irr_masks) - 1)
                mask = cv2.imread(self.irr_masks[mask_index],
                                  cv2.IMREAD_GRAYSCALE)
            elif rdv < self.config.mask_rates[1]:
                mask_index = random.randint(0, len(self.seg_masks) - 1)
                mask = cv2.imread(self.seg_masks[mask_index],
                                  cv2.IMREAD_GRAYSCALE)
            else:
                mask_index1 = random.randint(0, len(self.seg_masks) - 1)
                mask_index2 = random.randint(0, len(self.irr_masks) - 1)
                mask1 = cv2.imread(self.seg_masks[mask_index1],
                                   cv2.IMREAD_GRAYSCALE).astype(np.float)
                mask2 = cv2.imread(self.irr_masks[mask_index2],
                                   cv2.IMREAD_GRAYSCALE).astype(np.float)
                mask = np.clip(mask1 + mask2, 0, 255).astype(np.uint8)

            if mask.shape[0] != imgh or mask.shape[1] != imgw:
                mask = cv2.resize(mask, (imgw, imgh),
                                  interpolation=cv2.INTER_NEAREST)
            mask = (mask > 127).astype(np.uint8) * 255  # threshold due to interpolation
            return mask

    def to_tensor(self, img, norm=False):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        if norm:
            img_t = F.normalize(img_t, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        return img_t

    def resize(self, img, height, width, center_crop=False):
        imgh, imgw = img.shape[0:2]

        if center_crop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        if imgh > height and imgw > width:
            inter = cv2.INTER_AREA
        else:
            inter = cv2.INTER_LINEAR
        img = cv2.resize(img, (height, width), interpolation=inter)

        return img

    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist.sort(key=lambda x: x.split('/')[-1])
                return flist

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except:
                    return [flist]

        return []

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(dataset=self, batch_size=batch_size, drop_last=True, collate_fn=self.collate_fn)
            for item in sample_loader:
                yield item

    @staticmethod
    def collate_fn(batch):
        keys = batch[0].keys()
        res = {}
        for k in keys:
            temp_ = []
            for b in batch:
                if b[k] is not None:
                    temp_.append(b[k])
            if len(temp_) > 0:
                if type(temp_[0]) == str:
                    res[k] = temp_
                else:
                    res[k] = default_collate(temp_)
            else:
                res[k] = None

        return res


class ECDataset(torch.utils.data.Dataset):
    def __init__(self, config, flist, irr_mask_path=None, seg_mask_path=None,
                 fix_mask_path=None, training=True):
        super(ECDataset, self).__init__()
        self.config = config
        self.training = training
        self.data = self.load_flist(flist)
        self.irr_masks = []
        self.seg_masks = []
        self.fix_masks = []
        self.hawp_height = 512
        self.hawp_width = 512
        if irr_mask_path is not None:
            self.irr_masks = self.load_flist(irr_mask_path)
        if seg_mask_path is not None:
            self.seg_masks = self.load_flist(seg_mask_path)
        if fix_mask_path is not None:
            self.fix_masks = self.load_flist(fix_mask_path)
        if not training:
            assert len(self.fix_masks) == len(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.load_item(index)

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    def load_edge(self, img):

        return canny(img, sigma=2, mask=None).astype(np.float)

    def load_item(self, index):

        size = self.config.input_size
        fname = os.path.basename(self.data[index]).split('.')[0]
        # load image
        img = cv2.imread(self.data[index])
        while img is None:
            print('Bad image {}...'.format(self.data[index]))
            index = random.randint(0, len(self.data) - 1)
            img = cv2.imread(self.data[index])
        img = img[:, :, ::-1]

        # resize/crop if needed
        img = self.resize(img, size, size, center_crop=self.config.center_crop)
        # load mask
        mask = self.load_mask(img, index)
        # augment mask
        if self.training is True and self.config.flip is True:
            if random.random() < 0.5:
                mask = mask[:, ::-1, ...]
            if random.random() < 0.5:
                mask = mask[::-1, :, ...]

        # augment data
        if self.training is True and self.config.flip is True:
            if random.random() < 0.5:
                img = img[:, ::-1, ...]

        img_gray = rgb2gray(img)
        edge = self.load_edge(img_gray)

        img = self.to_tensor(img, norm=True)  # norm to -1~1
        edge = self.to_tensor(edge)
        mask = self.to_tensor(mask)
        meta = {'img': img, 'mask': mask, 'edge': edge, 'name': os.path.basename(self.data[index])}

        return meta

    def load_mask(self, img, index):
        imgh, imgw = img.shape[0:2]

        # test mode: load mask non random
        if self.training is False:
            mask = cv2.imread(self.fix_masks[index], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (imgw, imgh),
                              interpolation=cv2.INTER_NEAREST)
            mask = (mask > 127).astype(np.uint8) * 255
            return mask
        else:  # train mode: 40% mask with random brush, 40% mask with coco mask, 20% with additions
            rdv = random.random()
            if rdv < self.config.mask_rates[0]:
                mask_index = random.randint(0, len(self.irr_masks) - 1)
                mask = cv2.imread(self.irr_masks[mask_index],
                                  cv2.IMREAD_GRAYSCALE)
            elif rdv < self.config.mask_rates[1]:
                mask_index = random.randint(0, len(self.seg_masks) - 1)
                mask = cv2.imread(self.seg_masks[mask_index],
                                  cv2.IMREAD_GRAYSCALE)
            else:
                mask_index1 = random.randint(0, len(self.seg_masks) - 1)
                mask_index2 = random.randint(0, len(self.irr_masks) - 1)
                mask1 = cv2.imread(self.seg_masks[mask_index1],
                                   cv2.IMREAD_GRAYSCALE).astype(np.float)
                mask2 = cv2.imread(self.irr_masks[mask_index2],
                                   cv2.IMREAD_GRAYSCALE).astype(np.float)
                mask = np.clip(mask1 + mask2, 0, 255).astype(np.uint8)

            if mask.shape[0] != imgh or mask.shape[1] != imgw:
                mask = cv2.resize(mask, (imgw, imgh),
                                  interpolation=cv2.INTER_NEAREST)
            mask = (mask > 127).astype(np.uint8) * 255  # threshold due to interpolation
            return mask

    def to_tensor(self, img, norm=False):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        if norm:
            img_t = F.normalize(img_t, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        return img_t

    def resize(self, img, height, width, center_crop=False):
        imgh, imgw = img.shape[0:2]

        if center_crop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        if imgh > height and imgw > width:
            inter = cv2.INTER_AREA
        else:
            inter = cv2.INTER_LINEAR
        img = cv2.resize(img, (height, width), interpolation=inter)

        return img

    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist.sort(key=lambda x: x.split('/')[-1])
                return flist

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except:
                    return [flist]

        return []

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(dataset=self, batch_size=batch_size, drop_last=True, collate_fn=self.collate_fn)
            for item in sample_loader:
                yield item

    @staticmethod
    def collate_fn(batch):
        keys = batch[0].keys()
        res = {}
        for k in keys:
            temp_ = []
            for b in batch:
                if b[k] is not None:
                    temp_.append(b[k])
            if len(temp_) > 0:
                if type(temp_[0]) == str:
                    res[k] = temp_
                else:
                    res[k] = default_collate(temp_)
            else:
                res[k] = None

        return res
