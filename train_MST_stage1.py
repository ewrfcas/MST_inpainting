import argparse
import os
import random
import time
from shutil import copyfile

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataloader import LSMDataset
from src.lsm_hawp.detector import WireframeDetector, hawp_inference_test
from src.metrics import EdgeAccuracy
from src.models import SharedWEModel
from src.training import backward, get_opt, convert_fp16, save_model, load_model
from utils.logger import setup_logger
from utils.utils import Config, Progbar, to_cuda, postprocess, stitch_images, torch_show_all_params

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help='model checkpoints path')
    parser.add_argument('--config', type=str, required=True, help='model config path')
    parser.add_argument('--gpu', type=str, required=True, help='gpu ids')

    args = parser.parse_args()
    args.path = os.path.join('check_points', args.path)
    config_path = os.path.join(args.path, 'config.yml')

    # create checkpoints path if does't exist
    if not os.path.exists(args.path):
        os.makedirs(args.path)

    # copy config template if does't exist
    if not os.path.exists(config_path):
        copyfile('./{}'.format(args.config), config_path)

    # load config file
    config = Config(config_path)
    config.path = args.path
    config.gpu_ids = args.gpu

    # cuda visble devices
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_ids
    n_gpu = torch.cuda.device_count()

    # init device
    if torch.cuda.is_available():
        config.device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True  # cudnn auto-tuner
    else:
        config.device = torch.device("cpu")

    log_file = 'log-{}.txt'.format(time.time())
    logger = setup_logger(os.path.join(args.path, 'logs'), logfile_name=log_file)
    for k in config._dict:
        logger.info("{}:{}".format(k, config._dict[k]))

    # save samples and eval pictures
    os.makedirs(os.path.join(args.path, 'samples_stage1'), exist_ok=True)

    # set cv2 running threads to 1 (prevents deadlocks with pytorch dataloader)
    cv2.setNumThreads(0)

    # initialize random seed
    seed = config.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    # set dataset
    train_dataset = LSMDataset(config, config.data_flist[config.dataset]['train'],
                               wireframe_path=config.data_flist[config.dataset]['wireframe_path'],
                               irr_mask_path=config.irregular_path, seg_mask_path=config.train_seg_path,
                               wireframe_mask_rate=config.wireframe_mask_rate, hawp_th=config.hawp_th,
                               training=True)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        num_workers=12,
        shuffle=True,
        drop_last=True,
        collate_fn=train_dataset.collate_fn
    )
    val_dataset = LSMDataset(config, config.data_flist[config.dataset]['val'],
                             fix_mask_path=config.data_flist[config.dataset]['test_mask'],
                             training=False)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.batch_size,
        num_workers=4,
        drop_last=False,
        shuffle=False,
        collate_fn=val_dataset.collate_fn
    )
    sample_iterator = val_dataset.create_iterator(config.sample_size)

    model = SharedWEModel(config, input_channel=6, image_output_channel=3)
    lsm_hawp = WireframeDetector(is_cuda=True if str(config.device) != 'cpu' else False)
    lsm_hawp = lsm_hawp.to(config.device)
    lsm_hawp.load_state_dict(torch.load(config.lsm_hawp_ckpt)['model'])
    hawp_mean = torch.tensor([109.730, 103.832, 98.681]).to(config.device).reshape(1, 3, 1, 1)
    hawp_std = torch.tensor([22.275, 22.124, 23.229]).to(config.device).reshape(1, 3, 1, 1)
    g_opt, d_opt, g_sche, d_sche = get_opt(config, model, drop_steps=config.drop_steps_stage1)
    if config.float16:
        model, g_opt, d_opt, amp = convert_fp16(model, g_opt, d_opt)
    else:
        amp = None
    model, g_opt, d_opt, g_sche, d_sche, amp = load_model(model, g_opt, d_opt, g_sche, d_sche, amp=amp)
    steps_per_epoch = len(train_dataset) // config.batch_size
    iteration = model.iteration
    epoch = model.iteration // steps_per_epoch
    logger.info('Generator Parameters:{}'.format(torch_show_all_params(model.g_model)))
    logger.info('Discriminator Parameters:{}'.format(torch_show_all_params(model.d_model)))
    logger.info('Ngpu:{}'.format(n_gpu))
    logger.info('Start from epoch:{}, iteration:{}'.format(epoch, iteration))

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    model.train()
    keep_training = True
    best_f1 = 0
    edge_metric = EdgeAccuracy(threshold=0.5)
    best_iteration = 0
    while (keep_training):
        epoch += 1

        stateful_metrics = ['epoch', 'iter', 'g_lr']
        progbar = Progbar(len(train_dataset), max_iters=steps_per_epoch,
                          width=20, stateful_metrics=stateful_metrics)

        for items in train_loader:
            model.train()
            items = to_cuda(items, config.device)
            g_loss, d_loss, logs = model.process(items['img'], items['line'], items['edge'],
                                                 items['mask'], items['real_line'])
            if n_gpu > 1:
                g_loss = g_loss.mean()
                d_loss = d_loss.mean()
            g_opt, d_opt, g_sche, d_sche = backward(g_opt, d_opt, g_sche, d_sche, g_loss, d_loss,
                                                    float16=config.float16, amp=amp)
            iteration += 1

            logs = [("epoch", epoch), ("iter", iteration), ('g_lr', g_sche.get_lr()[0])] + logs
            progbar.add(config.batch_size, values=logs)

            if iteration % config.log_iters == 0:
                logger.debug(str(logs))
            if iteration % config.sample_iters == 0:
                model.eval()
                with torch.no_grad():
                    items = next(sample_iterator)
                    items = to_cuda(items, config.device)
                    # for inference, use lines output from lsm-hawp
                    items['line'] = hawp_inference_test(lsm_hawp, items['line'], items['mask'],
                                                        hawp_mean, hawp_std, config.device,
                                                        config.input_size, obj_remove=False, mask_th=config.hawp_th)
                    outputs = model(items['img'], items['line'], items['edge'], items['mask'])
                    show_results = [postprocess(outputs['images_masked']),
                                    postprocess(outputs['edges_masked'], simple_norm=True),
                                    postprocess(outputs['lines_masked'], simple_norm=True),
                                    postprocess(outputs['img_out'][-1]),
                                    postprocess(outputs['edge_out'][-1], simple_norm=True),
                                    postprocess(outputs['line_out'][-1], simple_norm=True)]
                    images = stitch_images(postprocess(items['img']), show_results, img_per_row=1)
                sample_name = os.path.join(args.path, 'samples_stage1', str(iteration).zfill(7) + ".jpg")

                print('\nsaving sample {}\n'.format(sample_name))
                images.save(sample_name)

            if iteration % config.eval_iters == 0:
                model.eval()
                eval_progbar = Progbar(len(val_dataset), width=20)
                index = 0
                edge_ps, edge_rs, edge_f1s = [], [], []
                line_ps, line_rs, line_f1s = [], [], []
                with torch.no_grad():
                    for items in tqdm(val_loader):
                        items = to_cuda(items, config.device)
                        # for inference, use lines output from lsm-hawp
                        items['line'] = hawp_inference_test(lsm_hawp, items['line'], items['mask'],
                                                            hawp_mean, hawp_std, config.device,
                                                            config.input_size, obj_remove=False, mask_th=config.hawp_th)
                        real_lines = hawp_inference_test(lsm_hawp, items['line'], torch.zeros_like(items['mask']),
                                                        hawp_mean, hawp_std, config.device,
                                                        config.input_size, obj_remove=False, mask_th=config.hawp_th)
                        outputs = model(items['img'], items['line'], items['edge'], items['mask'])

                        edge_p, edge_r, edge_f1 = edge_metric(items['edge'], outputs['edge_out'][-1])
                        line_p, line_r, line_f1 = edge_metric(real_lines, outputs['line_out'][-1])

                        edge_ps.append(edge_p.item())
                        edge_rs.append(edge_r.item())
                        edge_f1s.append(edge_f1.item())

                        line_ps.append(line_p.item())
                        line_rs.append(line_r.item())
                        line_f1s.append(line_f1.item())

                edge_p_score = np.mean(edge_ps)
                edge_r_score = np.mean(edge_rs)
                edge_f1_score = np.mean(edge_f1s)

                line_p_score = np.mean(line_ps)
                line_r_score = np.mean(line_rs)
                line_f1_score = np.mean(line_f1s)

                average_f1 = (edge_f1_score + line_f1_score) / 2

                logger.info('\nEdge_P:{0:.3f}, Edge_R:{1:.3f}, Edge_F1:{2:.3f}, '
                            'Line_P:{3:.3f}, Line_R:{4:.3f}, Line_F1:{5:.3f}, AF1:{6:.3f}\n'.
                            format(edge_p_score, edge_r_score, edge_f1_score, line_p_score,
                                   line_r_score, line_f1_score, average_f1))

                if config.save_best:
                    if best_f1 < average_f1:
                        best_f1 = average_f1
                        best_iteration = iteration
                        save_model(model_without_ddp, prefix='best_f1', g_opt=g_opt, d_opt=d_opt,
                                   amp=amp, iteration=iteration)

            if iteration % config.save_iters == 0:
                save_model(model_without_ddp, prefix='last', g_opt=g_opt, d_opt=d_opt,
                           amp=amp, iteration=iteration)

            if iteration >= config.max_iters_stage1:
                keep_training = False
                break

    logger.info('Best F1: {}, Iteration: {}'.format(best_f1, best_iteration))
