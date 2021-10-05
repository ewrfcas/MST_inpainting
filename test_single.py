import argparse
import os
import random

import numpy as np
import torch

from src.model_inference import MST
from utils.utils import Config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True, help='image path')
    parser.add_argument('--mask_path', type=str, required=True, help='mask path')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--PATH', type=str, default='./check_points/MST_P2M',
                        help='MST_P2M:Man-made Places2, MST_P2C:Comprehensive Places2, '
                             'MST_shanghaitech:all man-made scenes')
    parser.add_argument('--valid_th', type=float, default=0.85)
    parser.add_argument('--mask_th', type=float, default=0.8)
    parser.add_argument('--not_obj_remove', action='store_true', default=False)
    parser.add_argument('--config_path', type=str, default='./config.yml', help='config path')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    # load config file
    config = Config(args.config_path)

    # test mode
    config.PATH = args.PATH
    config.valid_th = args.valid_th
    config.mask_th = args.mask_th

    # init device
    if torch.cuda.is_available():
        config.DEVICE = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        config.DEVICE = torch.device("cpu")

    print('DEVICE:', config.DEVICE)

    # initialize random seed
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    concat_mask = True if 'P2C' in args.PATH else False
    model = MST(config, concat_mask)
    model.load()
    model.inference(args.image_path, args.mask_path, config.valid_th, config.mask_th,
                    not_obj_remove=args.not_obj_remove)
