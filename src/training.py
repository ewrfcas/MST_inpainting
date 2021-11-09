import os

import torch

try:
    from apex import amp

    amp.register_float_function(torch, 'matmul')
except ImportError:
    print('No apex for fp16...')


# combine image with origin images
def image_combine(source, target, mask):
    res = source * (1 - mask) + target * mask  # [b,3,h,w]
    return res


def load_model(model, amp=None):
    g_path = model.g_path + '_last.pth'
    if model.config.restore:
        if os.path.exists(g_path):
            print('Loading %s generator...' % g_path)
            data = torch.load(g_path, map_location='cpu')
            model.g_model.load_state_dict(data['g_model'])
            if model.config.restore:
                model.g_opt.load_state_dict(data['g_opt'])
            model.iteration = data['iteration']
            # for _ in range(model.iteration):
            #     g_sche.step()
            #     d_sche.step()
        else:
            print(g_path, 'not Found')
            raise FileNotFoundError

    d_path = model.d_path + '_last.pth'
    if model.config.restore:  # D is only use for training
        if os.path.exists(d_path):
            print('Loading %s discriminator...' % d_path)
            data = torch.load(d_path, map_location='cpu')
            model.d_model.load_state_dict(data['d_model'])
            if model.config.restore:
                model.d_opt.load_state_dict(data['d_opt'])
        else:
            print(d_path, 'not Found')
            raise FileNotFoundError
        if amp is not None and model.config.float16 and 'amp' in data:
            amp.load_state_dict(data['amp'])
    else:
        print('No need for discriminator during testing')

    return model, amp


def save_model(model, prefix=None, g_opt=None, d_opt=None, amp=None, iteration=None, n_gpu=1):
    if prefix is not None:
        save_g_path = model.g_path + "_{}.pth".format(prefix)
        save_d_path = model.d_path + "_{}.pth".format(prefix)
    else:
        save_g_path = model.g_path + ".pth"
        save_d_path = model.d_path + ".pth"

    print('\nsaving {}...\n'.format(prefix))
    save_g = model.g_model.module if n_gpu > 1 else model.g_model
    save_d = model.d_model.module if n_gpu > 1 else model.d_model
    torch.save({'iteration': model.iteration if iteration is None else iteration,
                'g_model': save_g.state_dict(),
                'g_opt': g_opt.state_dict(),
                'amp': amp.state_dict() if amp is not None else None},
               save_g_path)

    torch.save({'d_model': save_d.state_dict(),
                'd_opt': d_opt.state_dict()},
               save_d_path)
