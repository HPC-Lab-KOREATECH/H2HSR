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


import math
import time
import random
import datetime
from tqdm import tqdm
from abc import ABC, abstractmethod

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter 

from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

import models
import modules
from utils import utils as utl
from datasets import MITdataset
from modules import HoloLoss, ASMLoss
import optics as opt

class BaseTrainer(ABC):
    def __init__(self, config, args):
        self.cfg = config
        self.resume = args.resume
        self.epoch = 0

        # params for ddp 
        self.is_ddp = args.ddp
        self.rank = None

        # ddp setup
        self.init_for_ddp(args)
        
        # train modules
        self.model = None
        self.optimizer = None
        self.scheduler = None

    def init_for_ddp(self, args):
        if self.is_ddp:
            utl.init_distributed_mode(args)
            self.rank = utl.get_rank()
            self.cfg.num_gpus = utl.get_world_size()

            if utl.is_main_process():
                print(f'[Info] Multi GPUs for training are ready!')
            print(f'[Info] Rank: {self.rank}')
        else:
            self.rank = 0
            self.cfg.num_gpus = 1
            print(f'[Info] Single GPU is ready to train!')

    @abstractmethod
    def init_for_module(self):
        pass

    @abstractmethod
    def init_for_resume(self):
        pass

    @abstractmethod
    def init_for_dataloader(self):
        pass

    @abstractmethod
    def run(self):
        pass

class H2HSRTrainer(BaseTrainer):
    def __init__(self, config, args):
        super().__init__(config, args)

        self.dataloader = {'train': None, 'valid': None}
        self.sampler    = {'train': None, 'valid': None}

        # modules
        self.ASM = None
        self.criterion = None
        self.metrics = None
        self.weights = { 'holo' : 1.0, 'asm' : 1.0, 'tv' : 1.0 }

        # log
        self.best_logs = None
        self.loss = {
            'holo'          : 0.0,
            'asm'           : 0.0,
            'tv'            : 0.0    
        }
        self.logs = {
            'total'         : 0.0,
            'holo'          : 0.0,
            'asm'           : 0.0,
            'tv'            : 0.0,        
            'recon_psnr'    : 0.0,
            'recon_ssim'    : 0.0
        }
        self.writer = None

    def init_for_module(self):
        self.model = models.make(self.cfg.model_name).to(self.rank)
        if self.is_ddp:
            self.model = DDP(self.model, device_ids=[self.rank])
        self.optimizer = modules.load_optimizer(self.model, self.cfg.optimizer)
        self.scheduler = modules.load_scheduler(self.optimizer, self.cfg.scheduler)

    def init_for_resume(self):
        if self.resume:
            if self.cfg.ckp_path is None:
                raise ValueError(f'[Error] Model checkpoint is missing to resume training!')
            else:
                checkpoint = torch.load(os.path.join(self.cfg.ckp_path, self.cfg.model_name + '.pt'))
                self.model.load_state_dict(utl.remove_ddp_prefix(checkpoint['model_state_dict']))
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.scheduler.load_state_dict(checkpoint['scheduler'])
                self.epoch = checkpoint['epoch']
        else:
            if utl.is_main_process():
                print(f'[Info] The training is starting from the scratch.')

    def load_dataloader(self, mode='train'):
        dataset = MITdataset(self.cfg.root, mode)
        sampler = DistributedSampler(dataset, utl.get_world_size(), utl.get_rank(), shuffle=True)
        dataloader = DataLoader(dataset,
                                batch_size=self.cfg.batch_size // self.cfg.num_gpus,
                                shuffle=(sampler is None) if mode == 'train' else False,
                                num_workers=self.cfg.num_workers,
                                pin_memory=True,
                                sampler=sampler)
        return dataloader, sampler

    def init_for_dataloader(self):
        # Generation dataloader and sampler
        for mode in ['train', 'valid']:
            self.dataloader[mode], self.sampler[mode] = self.load_dataloader(mode)

    def feed_data(self, data):
        lr = data['lr'].to(self.rank, non_blocking=True)
        hr = data['hr'].to(self.rank, non_blocking=True)
        depth = data['depth'].to(self.rank, non_blocking=True)
        return lr, hr, depth

    def inference(self, lr):
        sr = self.model(lr)
        # normalization: [-1,1] -> [0,1]
        sr = sr * 0.5 + 0.5
        return sr
    
    def calculate_loss(self, sr, hr, depth):
        # data fidelty loss: hologram
        self.loss['holo'] = self.criterion['holo'](sr, hr)

        # perceptual loss: reconstruction
        self.loss['asm'], self.loss['tv'] = self.criterion['asm'](sr, hr, depth)

    def save_checkpoint(self, save_path, save_name):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        torch.save({
            'epoch': self.epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }, os.path.join(save_path, save_name))

    def train(self):
        self.model.train()

        with tqdm(self.dataloader['train'], unit="batch", leave=False) as T:
            for idx, data in enumerate(T):
                t_start = time.time()
                self.optimizer.zero_grad()

                lr, hr, depth = self.feed_data(data)
                sr = self.inference(lr)
                self.calculate_loss(sr, hr, depth)

                total_loss = 0.0
                for (wk, wv), (lk, lv) in zip(self.weights.items(), self.loss.items()):
                    total_loss += wv * lv
                
                total_loss.backward()
                self.optimizer.step()

                # record train info into tensorboard
                if idx % self.cfg.print_freq == 0:
                    num_data = len(self.dataloader['train'])
                    iters = idx + self.epoch * num_data + 1

                    self.loss['total'] = total_loss.item()

                    if utl.is_main_process():
                        self.writer.add_scalar("Train/Loss", self.loss['total'], iters)
                        self.writer.add_scalar("Train/Holo", self.loss['holo'],  iters)
                        self.writer.add_scalar("Train/ASM",  self.loss['asm'],   iters)
                        self.writer.add_scalar("Train/TV",   self.loss['tv'],    iters)

                        utl.print_all_info_log(self.loss, 
                                            self.weights, 
                                            self.epoch, 
                                            idx,
                                            num_data,
                                            self.cfg)
                        utl.print_progress(start=t_start, 
                                           size=num_data, 
                                           index=idx,
                                           epochs=self.cfg.epochs,
                                           epoch=self.epoch)

    def valid(self):
        self.model.eval()

        # reset log value 
        for k, v in self.logs.items():
            self.logs[k] = 0.0

        with torch.no_grad():
            with tqdm(self.dataloader['valid'], unit="batch", leave=False) as T:
                for data in T:
                    T.set_description(f"Epoch {self.epoch}")

                    lr, hr, depth = self.feed_data(data)
                    sr = self.inference(lr)
                    self.calculate_loss(sr, hr, depth)

                    total_loss = 0.0
                    for (wk, wv), (lk, lv) in zip(self.weights.items(), self.loss.items()):
                        total_loss += wv * lv
                    
                    self.logs['total'] += total_loss.item()
                    self.logs['holo']  += self.loss['holo']
                    self.logs['asm']   += self.loss['asm']
                    self.logs['tv']    += self.loss['tv']

                    # evaluate the quality of reconstruction
                    z = (random.uniform(self.cfg.z[0], self.cfg.z[1]))
                    recon_sr = opt.reconstruction(sr, z, self.ASM)
                    recon_hr = opt.reconstruction(hr, z, self.ASM)
                    
                    # PSNR & SSIM
                    self.logs['recon_psnr'] += self.metrics['recon_psnr'](recon_sr, recon_hr);
                    self.logs['recon_ssim'] += self.metrics['recon_ssim'](recon_sr, recon_hr);

                    T.set_postfix(loss=total_loss.item())

        # divided by number of batches 
        size = len(self.dataloader['valid'])
        for key, value in self.logs.items():
            self.logs[key] = value / size
        
        # record the log
        if utl.is_main_process():
            self.writer.add_scalar("Valid/Loss",           self.logs['total'],       self.epoch) 
            self.writer.add_scalar("Valid/Holo",           self.logs['holo'],        self.epoch)
            self.writer.add_scalar("Valid/ASM",            self.logs['asm'],         self.epoch)
            self.writer.add_scalar("Valid/TV",             self.logs['tv'],          self.epoch)
            self.writer.add_scalar("Valid/ReconPSNR",      self.logs['recon_psnr'],  self.epoch)
            self.writer.add_scalar("Valid/ReconSSIM",      self.logs['recon_ssim'],  self.epoch)
            
            if self.epoch % 10 == 0:
                utl.save_sample_images(sr, hr, 2, self.cfg.save_path, self.epoch, mode='holo')
                utl.save_sample_images(recon_sr, recon_hr, 1, self.cfg.save_path, self.epoch, mode='recon')

        self.scheduler.step()

    def run(self):
        # load modules
        self.init_for_module()
        
        # resuming check
        self.init_for_resume()
        
        # load dataloader
        self.init_for_dataloader()

        # setup diffraction method
        self.ASM = opt.AngularSpectrumMethod(
            input_size  = self.cfg.data_size['hr'],
            pixel_pitch = self.cfg.pixelpitch,
            wavelength  = self.cfg.wavelength,
            device      = self.rank 
        )

        # criterion, weights, metrics
        self.criterion = {
            'holo': HoloLoss().to(self.rank),
            'asm' : ASMLoss(self.ASM, self.cfg.params).to(self.rank)
        }
        self.metrics = { 'recon_psnr' : PSNR(data_range=self.cfg.max_val).to(self.rank), 
                         'recon_ssim' : SSIM(data_range=self.cfg.max_val).to(self.rank)}

        # tensorboard logger
        if utl.is_main_process():
            self.writer = SummaryWriter()
            self.cfg.save_path = utl.get_save_path(root=self.cfg.save_root)
            self.best_logs = self.logs.copy()

        # train loop
        for epoch in range(self.cfg.epochs):
            self.sampler['train'].set_epoch(epoch);
            self.sampler['valid'].set_epoch(epoch);
            self.epoch = epoch

            # train
            self.train()

            # valid
            self.valid()

            # compare with best logs and save checkpoints
            utl.reduce_logs(self.logs)

            if utl.is_main_process():
                if utl.check_logger(self.best_logs, self.logs):
                    self.save_checkpoint(self.cfg.save_path, self.cfg.model_name + '.pt')
                    self.best_logs = self.logs.copy()

                if self.epoch % self.cfg.save_freq == 0 or self.epoch == self.cfg.epochs:
                    self.save_checkpoint(self.cfg.save_path, self.cfg.model_name + f'_ep{self.epoch}.pt')

