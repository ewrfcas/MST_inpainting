import argparse
import os
import random
import time
from shutil import copyfile

import cv2
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from tqdm import tqdm

from src.dataloader import LSMDataset
from src.lsm_hawp.detector import WireframeDetector, hawp_inference_test
from src.metrics import get_inpainting_metrics
from src.models import InpaintingModel, SharedWEModel
from src.training import save_model, load_model, image_combine
from utils.logger import setup_logger
from utils.utils import Config, Progbar, to_cuda, postprocess, stitch_images, torch_show_all_params, imsave

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help='model checkpoints path')
    parser.add_argument('--config', type=str, required=True, help='model config path')
    parser.add_argument('--gpu', type=str, required=True, help='gpu ids')
    parser.add_argument('--local_rank', type=int, default=-1, help='the id of this machine')
    parser.add_argument('--nodes', type=int, default=1, help='how many machines')

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
    if n_gpu > 1:
        config.world_size = args.nodes * n_gpu
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12382'

        dist.init_process_group(backend="nccl")
        local_rank = torch.distributed.get_rank()
    else:
        config.world_size = 1
        local_rank = 0

    # init device
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        config.device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True  # cudnn auto-tuner
    else:
        config.device = torch.device("cpu")

    if local_rank == 0:
        log_file = 'log-{}.txt'.format(time.time())
        logger = setup_logger(os.path.join(args.path, 'logs'), logfile_name=log_file)
        for k in config._dict:
            logger.info("{}:{}".format(k, config._dict[k]))

        # save samples and eval pictures
        os.makedirs(os.path.join(args.path, 'samples_stage2'), exist_ok=True)
        os.makedirs(os.path.join(args.path, 'eval_stage2'), exist_ok=True)
    else:
        logger = None

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
    if n_gpu > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=config.world_size,
                                           rank=local_rank, shuffle=True)
    else:
        train_sampler = RandomSampler(train_dataset)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size // config.world_size,
        num_workers=12,
        drop_last=True,
        sampler=train_sampler,
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

    model = InpaintingModel(config, input_channel=7).to(config.device)
    model, amp = load_model(model, amp=None)
    # load stage1 model
    model_stage1 = SharedWEModel(config, input_channel=6, image_output_channel=3).to(config.device)
    model_stage1.g_model.load_state_dict(torch.load(model_stage1.g_path + '_last.pth',
                                                    map_location='cpu')['g_model'])
    model_stage1.eval()
    lsm_hawp = WireframeDetector(is_cuda=True if str(config.device) != 'cpu' else False)
    lsm_hawp = lsm_hawp.to(config.device)
    lsm_hawp.load_state_dict(torch.load(config.lsm_hawp_ckpt, map_location='cpu')['model'])
    hawp_mean = torch.tensor([109.730, 103.832, 98.681]).to(config.device).reshape(1, 3, 1, 1)
    hawp_std = torch.tensor([22.275, 22.124, 23.229]).to(config.device).reshape(1, 3, 1, 1)
    steps_per_epoch = len(train_dataset) // config.batch_size
    iteration = model.iteration
    epoch = model.iteration // steps_per_epoch
    if local_rank == 0:
        logger.info('Generator Parameters:{}'.format(torch_show_all_params(model.g_model)))
        logger.info('Discriminator Parameters:{}'.format(torch_show_all_params(model.d_model)))
        logger.info('Ngpu:{}'.format(n_gpu))
        logger.info('Start from epoch:{}, iteration:{}'.format(epoch, iteration))

    if n_gpu > 1:
        if config.float16:
            from apex.parallel import DistributedDataParallel as DDP

            model.g_model = DDP(model.g_model)
            model.d_model = DDP(model.d_model)
        else:
            from torch.nn.parallel import DistributedDataParallel as DDP

            model.g_model = DDP(model.g_model, device_ids=[local_rank], output_device=local_rank)
            model.d_model = DDP(model.d_model, device_ids=[local_rank], output_device=local_rank)

    model.train()
    keep_training = True
    best_fid = 9999
    best_iteration = 0
    while (keep_training):
        epoch += 1
        if n_gpu > 1:
            train_sampler.set_epoch(epoch)  ## Shuffle each epoch

        stage = 3 if iteration >= config.max_iters_stage2 else 2

        stateful_metrics = ['stage', 'epoch', 'iter', 'g_lr']
        progbar = Progbar(len(train_dataset) // config.world_size, max_iters=steps_per_epoch,
                          width=20, stateful_metrics=stateful_metrics, verbose=1 if local_rank == 0 else 0)

        for items in train_loader:
            model.train()
            items = to_cuda(items, config.device)

            if stage == 2:  # stage2, use GT edges, lines
                edges = items['edge']
                lines = items['real_line']
            else:  # stage3, use edges, lines from stage1
                with torch.no_grad():
                    outputs = model_stage1.forward(items['img'], items['line'], items['edge'], items['mask'])
                edges = outputs['edge_out'][-1]
                lines = outputs['line_out'][-1]

            _, g_loss, d_loss, logs = model.process(items['img'], edges, lines, items['mask'])
            iteration += 1

            logs = [("stage", stage), ("epoch", epoch), ("iter", iteration),
                    ('g_lr', model.g_sche.get_lr()[0])] + logs
            progbar.add(config.batch_size // config.world_size, values=logs)

            if iteration % config.log_iters == 0 and local_rank == 0:
                logger.debug(str(logs))
            if (iteration % config.sample_iters == 0 or iteration == 1) and local_rank == 0:
                model.eval()
                with torch.no_grad():
                    items = next(sample_iterator)
                    items = to_cuda(items, config.device)
                    edges = items['edge']
                    if stage == 2:  # stage2 use GT edges, lines
                        temp_mask = torch.zeros_like(items['mask'])  # all zero mask for GT lines
                    else:
                        temp_mask = items['mask']
                    lines = hawp_inference_test(lsm_hawp, items['line'], temp_mask, hawp_mean,
                                                hawp_std, config.device, config.input_size,
                                                obj_remove=False, mask_th=config.hawp_th)
                    if stage == 3:  # for stage3, use edges lines from stage1
                        outputs = model_stage1.forward(items['img'], lines, edges, items['mask'])
                        edges = outputs['edge_out'][-1]
                        lines = outputs['line_out'][-1]
                    edge_line_maps = torch.clamp(edges + lines, 0, 1.0)
                    infos = torch.cat([edges, lines, edge_line_maps], dim=1)
                    outputs = model(items['img'], infos, items['mask'])
                    show_results = [postprocess(items['img'] * (1 - items['mask']).float() + items['mask']),
                                    postprocess(edges, simple_norm=True),
                                    postprocess(lines, simple_norm=True),
                                    postprocess(edge_line_maps, simple_norm=True),
                                    postprocess(outputs)]
                    images = stitch_images(postprocess(items['img']), show_results, img_per_row=1)
                sample_name = os.path.join(args.path, 'samples_stage2', str(iteration).zfill(7) + ".jpg")

                print('\nsaving sample {}\n'.format(sample_name))
                images.save(sample_name)

            if (iteration % config.eval_iters == 0 or iteration == 1) and local_rank == 0:
                model.eval()
                eval_progbar = Progbar(len(val_dataset), width=20)
                index = 0
                with torch.no_grad():
                    for items in tqdm(val_loader):
                        items = to_cuda(items, config.device)
                        # for inference, use lines output from lsm-hawp
                        items['line'] = hawp_inference_test(lsm_hawp, items['line'], items['mask'],
                                                            hawp_mean, hawp_std, config.device,
                                                            config.input_size, obj_remove=False, mask_th=config.hawp_th)
                        outputs = model_stage1.forward(items['img'], items['line'], items['edge'], items['mask'])
                        edges = outputs['edge_out'][-1]
                        lines = outputs['line_out'][-1]
                        edge_line_maps = torch.clamp(edges + lines, 0, 1.0)
                        infos = torch.cat([edges, lines, edge_line_maps], dim=1)
                        fake_img = model(items['img'], infos, items['mask'])

                        fake_img = image_combine(items['img'], fake_img, items['mask'])
                        fake_img = postprocess(fake_img)  # [b, h, w, 3]
                        for i in range(fake_img.shape[0]):
                            sample_name = os.path.join(args.path, 'eval_stage2',
                                                       val_dataset.load_name(index)).replace('.jpg', '.png')
                            imsave(fake_img[i], sample_name)
                            index += 1

                        eval_progbar.add(fake_img.shape[0])

                score_dict = get_inpainting_metrics(config.data_flist[config.dataset]['test_res'],
                                                    os.path.join(args.path, 'eval_stage2'), logger, fid_test=True)

                if config.save_best:
                    if best_fid > score_dict['fid']:
                        best_fid = score_dict['fid']
                        best_iteration = iteration
                        save_model(model, prefix='best_fid', g_opt=model.g_opt, d_opt=model.d_opt,
                                   amp=None, iteration=iteration, n_gpu=n_gpu)

            if iteration % config.save_iters == 0 and local_rank == 0:
                save_model(model, prefix='last', g_opt=model.g_opt, d_opt=model.d_opt,
                           amp=None, iteration=iteration, n_gpu=n_gpu)

            if iteration >= config.max_iters_stage3:
                keep_training = False
                break

    if local_rank == 0:
        logger.info('Best FID: {}, Iteration: {}'.format(best_fid, best_iteration))
