import os

import torch
import torch.optim as optim

from utils.utils import get_lr_schedule_with_steps

try:
    from apex import amp

    amp.register_float_function(torch, 'matmul')
except ImportError:
    print('No apex for fp16...')


# combine image with origin images
def image_combine(source, target, mask):
    res = source * (1 - mask) + target * mask  # [b,3,h,w]
    return res


def model_inference(model, items):
    outputs = {}
    fake_img = model(items['img'], items['mask'])
    outputs['fake_img'] = fake_img
    return outputs


def backward(g_opt, d_opt, g_sche, d_sche, g_loss=None, d_loss=None,
             float16=False, amp=None, clip_norm=1.0):
    d_opt.zero_grad()
    if d_loss is not None:
        if float16:
            with amp.scale_loss(d_loss, d_opt, loss_id=0) as d_loss_scaled:
                d_loss_scaled.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(d_opt), clip_norm)
        else:
            d_loss.backward()
    d_opt.step()

    g_opt.zero_grad()
    if g_loss is not None:
        if float16:
            with amp.scale_loss(g_loss, g_opt, loss_id=1) as g_loss_scaled:
                g_loss_scaled.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(g_opt), clip_norm)
        else:
            g_loss.backward()
    g_opt.step()

    d_sche.step()
    g_sche.step()
    return g_opt, d_opt, g_sche, d_sche


def get_opt(config, model, drop_steps):
    g_opt = optim.Adam(params=model.g_model.parameters(),
                       lr=float(config.g_lr), betas=(config.beta1, config.beta2))
    d_opt = optim.Adam(params=model.d_model.parameters(),
                       lr=float(config.d_lr), betas=(config.beta1, config.beta2))
    g_sche = get_lr_schedule_with_steps(config.decay_type, g_opt, drop_steps=drop_steps, gamma=config.drop_gamma)
    d_sche = get_lr_schedule_with_steps(config.decay_type, d_opt, drop_steps=drop_steps, gamma=config.drop_gamma)

    return g_opt, d_opt, g_sche, d_sche


def get_ddp_opt(config, g_model, d_model):
    g_opt = optim.Adam(params=g_model.parameters(),
                       lr=float(config.g_lr), betas=(config.beta1, config.beta2))
    d_opt = optim.Adam(params=d_model.parameters(),
                       lr=float(config.d_lr), betas=(config.beta1, config.beta2))
    g_sche = get_lr_schedule_with_steps(config.decay_type, g_opt,
                                        drop_steps=config.drop_steps,
                                        gamma=config.drop_gamma)
    d_sche = get_lr_schedule_with_steps(config.decay_type, d_opt,
                                        drop_steps=config.drop_steps,
                                        gamma=config.drop_gamma)

    return g_opt, d_opt, g_sche, d_sche


def convert_fp16(model, g_opt, d_opt):
    model, [g_opt, d_opt] = amp.initialize(model, [g_opt, d_opt],
                                           num_losses=2, opt_level='O1')
    return model, g_opt, d_opt, amp


def convert_ddp_fp16(g_model, d_model, g_opt, d_opt):
    [g_model, d_model], [g_opt, d_opt] = amp.initialize([g_model, d_model], [g_opt, d_opt],
                                                        num_losses=2, opt_level='O1')
    return g_model, d_model, g_opt, d_opt, amp


def load_model(model, g_opt, d_opt, g_sche, d_sche, amp=None):
    g_path = model.g_path + '_last.pth'
    if model.config.restore:
        if os.path.exists(g_path):
            print('Loading %s generator...' % g_path)
            if torch.cuda.is_available():
                data = torch.load(g_path)
            else:
                data = torch.load(g_path, map_location=lambda storage, loc: storage)
            model.g_model.load_state_dict(data['g_model'])
            if model.config.restore:
                g_opt.load_state_dict(data['g_opt'])
            model.iteration = data['iteration']
            for _ in range(model.iteration):
                g_sche.step()
                d_sche.step()
        else:
            print(g_path, 'not Found')
            raise FileNotFoundError

    d_path = model.d_path + '_last.pth'
    if model.config.restore:  # D is only use for training
        if os.path.exists(d_path):
            print('Loading %s discriminator...' % d_path)
            if torch.cuda.is_available():
                data = torch.load(d_path)
            else:
                data = torch.load(d_path, map_location=lambda storage, loc: storage)
            model.d_model.load_state_dict(data['d_model'])
            if model.config.restore:
                d_opt.load_state_dict(data['d_opt'])
        else:
            print(d_path, 'not Found')
            raise FileNotFoundError
        if amp is not None and model.config.float16 and 'amp' in data:
            amp.load_state_dict(data['amp'])
    else:
        print('No need for discriminator during testing')

    return model, g_opt, d_opt, g_sche, d_sche, amp


def save_model(model, prefix=None, g_opt=None, d_opt=None, amp=None, iteration=None):
    if prefix is not None:
        save_g_path = model.g_path + "_{}.pth".format(prefix)
        save_d_path = model.d_path + "_{}.pth".format(prefix)
    else:
        save_g_path = model.g_path + ".pth"
        save_d_path = model.d_path + ".pth"

    print('\nsaving {}...\n'.format(prefix))
    torch.save({'iteration': model.iteration if iteration is None else iteration,
                'g_model': model.g_model.state_dict(),
                'g_opt': g_opt.state_dict(),
                'amp': amp.state_dict() if amp is not None else None},
               save_g_path)

    torch.save({'d_model': model.d_model.state_dict(),
                'd_opt': d_opt.state_dict()},
               save_d_path)


def save_ddp_model(g_model, g_path, d_model, d_path, prefix=None,
                   g_opt=None, d_opt=None, amp=None, iteration=None):
    if prefix is not None:
        save_g_path = g_path + "_{}.pth".format(prefix)
        save_d_path = d_path + "_{}.pth".format(prefix)
    else:
        save_g_path = g_path + ".pth"
        save_d_path = d_path + ".pth"

    print('\nsaving to {}...\n'.format(prefix))
    torch.save({'iteration': iteration,
                'g_model': g_model.state_dict(),
                'g_opt': g_opt.state_dict(),
                'amp': amp.state_dict() if amp is not None else None},
               save_g_path)

    torch.save({'d_model': d_model.state_dict(),
                'd_opt': d_opt.state_dict()},
               save_d_path)
