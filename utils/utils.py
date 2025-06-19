#
# Copyright (C) 2025, SPIN Lab, Korea University of Technology and Education (KOREATECH)
# and Digital Holography Research Group, Electronics and Telecommunications Research Institute (ETRI)
#
# This software is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).
# You may use, share, and adapt the material for non-commercial purposes, provided that appropriate credit is given.
# See the LICENSE.md file or visit https://creativecommons.org/licenses/by-nc/4.0/ for full terms.
#
# For inquiries, contact: bluekdct@gmail.com
#
# This code is based on the research published as:
# No, Y., Lee, J., Yeom, H., Kwon, S., and Kim, D.,
# "H2HSR: Hologram-to-Hologram Super-Resolution With Deep Neural Network," IEEE Access, 2024.
#


import os
import time
import torch
import datetime

import torch.distributed as dist
from torchvision.utils import save_image
from copy import deepcopy

def check_cuda():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #print(f'[Info] Using {device} device')
    return device

def remove_ddp_prefix(state_dict):
    # The ddp mode makes the layer name of model with 'module.'
    # Therefore, this function works to remove the prefix.
     
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            name = k[7:]
        else:
            name = k
        new_state_dict[name] = v
    return new_state_dict

def reduce_logs(logs):
    for key in logs:
        tensor = torch.tensor(logs[key]).float().cuda()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
        logs[key] = tensor.item()    

def get_save_path(root='./checkpoint', folder_name=None):    
    # set root of save folder
    if not os.path.exists(root):
        os.makedirs(root)
    
    # check folder_name
    if folder_name is None:
        date = str(datetime.datetime.now()).split('.')[0]
        date, time = date.split()[0], date.split()[1].replace(':', '-')
    
        save_path = os.path.join(root, date)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    
        save_path = os.path.join(save_path, time)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    else:
        save_path = os.path.join(folder_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        else:
            return -1
    
    return save_path

def get_result_path(root='./result/', model_name=None):
    if model_name is None:
        raise ValueError("[Error] There is no model name to create result folder.")

    if not os.path.exists(root):
        os.makedirs(root)
    
    save_path = os.path.join(root, model_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    date = str(datetime.datetime.now()).split('.')[0]
    date, time = date.split()[0], date.split()[1].replace(':', '-')

    save_path = os.path.join(save_path, date)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_path = os.path.join(save_path, time)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    return save_path

"""
    Data distributed process setup
"""
      
def is_main_process():
    return get_rank() == 0 or get_world_size() == 1

def save_on_master(*args, **kwargs):
    if is_main_process():
        save_model(*args, **kwargs)
        
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def enable_print(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}) (world {}): {}'.format(
        args.rank, args.world_size, 'env://'), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method='env://',
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()

    # enable_print(args.rank == 0)

"""
    Generate sample images during training
"""

def save_sample_images(y, _y, size, save_path, epoch, tag=None, mode='holo'):
    """
        Save the image of inference result
        - tag: if you hope to add the additional name into image
        - mode: 
            'holo' -> hologram
            'recon' -> reconstruction
    """
    path = os.path.join(save_path, 'output')
    if not os.path.exists(path):
        os.makedirs(path)

    if mode == 'holo': # hologram
        y_amp, y_phs = y[0:size, 0:3], y[0:size, 3:6]
        _y_amp, _y_phs = _y[0:size, 0:3], _y[0:size, 3:6]
        
        image = torch.cat([
            y_amp, y_phs, _y_amp, _y_phs
        ], dim=-1)

    elif mode == 'recon': # reconstruction
        y_recon, _y_recon = y[0:size], _y[0:size]

        image = torch.cat([
            y_recon, _y_recon
        ], dim=-1)

    else: # error
        raise ValueError(f'[Error] Not suitable tensor to make image.')
    
    if tag is not None:
        save_image(image, f'{path}/{mode}_{tag}_{epoch}.png', nrow=size, normalize=True)
    else:
        save_image(image, f'{path}/{mode}_{epoch}.png', nrow=size, normalize=True)


def check_logger(best, curr):
    state = []

    for (best_k, best_v), (curr_k, curr_v) in zip(best.items(), curr.items()):
        if 'recon' in best_k and 'recon' in curr_k:
            if best_v < curr_v: state.append(True)
            else: state.append(False)
    
    return all(state)

"""
    log print
"""

def print_all_info_log(logger, weights, epoch, batch_index, batch_size, config):
    """This function is to print all information (model info - name, weight factors,
       training info - loss names, values, PSNR, SSIM) in log

    Args:
        logger (_type_): Training logger dict. contains loss names and values and PSNR, SSIM
        weights (_type_): Training weights dict. contains loss names and factors
        epoch (_type_): Training epoch
        batch_index (_type_): Training batch index
        batch_size (_type_): Training batch size
        config (_type_): Project Configuration info
    """
    print_str = ""
    print_column = 2
    idx = 0
    epoch_info = f'{epoch} / {config.epochs}'
    batch_info = f'{batch_index} / {batch_size}'
    
    os.system('clear')
    print('------------------------------------------------------------------------------------------------')
    print('| Model Informations----------------------------------------------------------------------------')
    print(f'|{"Model":^15}|: {config.model_name:^50} ')
    for k, v in weights.items():
        print_str += f'|{k:^15}|: {v:<-10.4f} '
        idx += 1
        if idx % print_column == 0:
            print_str += '\n'
    print(print_str)

    print_str = ""
    idx = 0
    print('| Training Informations-------------------------------------------------------------------------')
    print(f'|{"Epoch":^15}|: {epoch_info:<10} |{"Batch":^15}|: {batch_info:<10}')
    for k, v in logger.items():
        print_str += f'|{k:^15}|: {v:<-10.6f} '
        idx += 1
        if idx % print_column == 0:
            print_str += '\n'
    print_str += '|\n------------------------------------------------------------------------------------------------'
    print(print_str)


def print_progress(start, size, index, epochs, epoch):
    print(f'|{"Each Batch Time":^15}|: {time.time() - start:<-.2f}s |{"Expected 1 Epoch Time":^15}|: {(time.time() - start) * (size - index):<-.2f}s|{"Expected All Epoch Time":^25}|: {str(datetime.timedelta(seconds=int(((time.time() - start) * (size - index)) + ((time.time() - start) * (size) * (epochs - epoch)))))}')