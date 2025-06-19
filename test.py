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
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1" # open for EXR file
import cv2
import math
import argparse

from tqdm import tqdm
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# evaluation metrics
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from modules import PhaseRotatePSNR

import models
from utils import utils as utl
from datasets import MITdataset
import optics as opt

class BaseTester(ABC):
    def __init__(self, config, args):
        self.cfg = config
        
        self.model = None
        self.dataloader = None
        self.metrics = None
    
    @abstractmethod
    def init_for_module(self):
        pass
    
    @abstractmethod
    def init_for_dataloader(self):
        pass

    @abstractmethod
    def quantitative(self):
        pass

    @abstractmethod
    def qualitative(self):
        pass
    
    @abstractmethod
    def run(self):
        pass

class H2HSRTester(BaseTester):
    def __init__(self, config, args):
        super().__init__(config, args)
        self.dev = 'cuda' # support only single GPU
        
        self.mode = args.eval_mode
        self.idx = args.index

        self.dataloader = {'test': None}
        self.amp_max = torch.tensor(self.cfg.amp_max).to(self.dev)
        self.ASM = None
        self.metrics = None
        self.logs = {
            'amp_psnr'      : 0.0,  # amplitude PSNR
            'amp_ssim'      : 0.0,  # amplitude SSIM
            'phs_psnr'      : 0.0,  # Phase PSNR
            'phs_ssim'      : 0.0,  # Phase SSIM
            'phs_rot_psnr'  : 0.0,  # Phase Rotation PSNR
            'recon_psnr'    : 0.0,  # Recon. PSNR
            'recon_ssim'    : 0.0   # Recon. SSIM
        }
        self.ckp_root = config.ckp_root

        if self.cfg.batch_size > 1: self.cfg.batch_size = 1

        # interpolation
        if args.model > 2:
            print("[Info] Test with bicubic interpolation!")
            self.is_inter = True
        else:
            self.is_inter = False

    def init_for_module(self):
        if self.is_inter:
            self.model = 'bicubic'
            self.cfg.model_name = 'bicubic'
        else:
            self.model = models.make(self.cfg.model_name).to(self.dev)            
            # load checkpoint
            checkpoint = torch.load(os.path.join(self.ckp_root, f'H2HSR_{self.cfg.model_name}.pt'))
            self.model.load_state_dict(utl.remove_ddp_prefix(checkpoint['model_state_dict']))
            self.model.eval()

    def load_dataloader(self, mode='test'):
        dataset = MITdataset(self.cfg.root, mode)
        dataloader = DataLoader(dataset,
                                batch_size=self.cfg.batch_size,
                                shuffle=False,
                                num_workers=self.cfg.batch_size * 2,
                                pin_memory=True)
        return dataloader

    def init_for_dataloader(self):
        self.dataloader = self.load_dataloader()

    def feed_data(self, data):
        lr = data['lr'].to(self.dev, non_blocking=True)
        hr = data['hr'].to(self.dev, non_blocking=True)
        return lr, hr

    def inference(self, lr):
        if self.is_inter:
            sr = F.interpolate(lr, scale_factor=(2, 2), mode=self.model)
        else:
            sr = self.model(lr)
            # normalization: [-1,1] -> [0,1]
            sr = sr * 0.5 + 0.5
        
        return sr

    def postprocessing(self, holo, phs_shift=False):
        amp, phs = holo[:, 0:3], holo[:, 3:6]
        amp *= self.amp_max.view(1, 3, 1, 1)
        
        if phs_shift:
            phs = phs - torch.mean(phs, dim=[2, 3], keepdims=True) + 0.5
        
        return torch.cat([amp, phs], dim=1)

    def quantitative(self):
        num_offsets = (self.cfg.z[1] - self.cfg.z[0]) / self.cfg.z_offset + 1

        with torch.no_grad():
            with tqdm(self.dataloader, unit="batch", leave=False) as T:
                for idx, data in enumerate(T):
                    lr, hr = self.feed_data(data)
                    sr = self.inference(lr)

                    sr_amp, sr_phs = sr[:, 0:3], sr[:, 3:6]
                    if not self.is_inter:
                        sr_phs = sr_phs - torch.mean(sr_phs, dim=[2, 3], keepdims=True) + 0.5
                    hr_amp, hr_phs = hr[:, 0:3], hr[:, 3:6]

                    self.logs["amp_psnr"] += self.metrics['amp_psnr'](sr_amp, hr_amp)
                    self.logs["amp_ssim"] += self.metrics['amp_ssim'](sr_amp, hr_amp)
                    self.logs["phs_psnr"] += self.metrics['phs_psnr'](sr_phs, hr_phs)
                    self.logs["phs_ssim"] += self.metrics['phs_ssim'](sr_phs, hr_phs)
                    self.logs["phs_rot_psnr"] += self.metrics['phs_rot_psnr'](sr_phs, hr_phs)

                    # reconstruction
                    shift = True if self.is_inter else False
                    sr = self.postprocessing(sr, phs_shift = shift)
                    hr = self.postprocessing(hr)

                    for idx_offset in range(int(num_offsets)):
                        z = self.cfg.z[0] + idx_offset * self.cfg.z_offset
                        
                        sr_recon = opt.reconstruction(sr, z, self.ASM['hr'])
                        hr_recon = opt.reconstruction(hr, z, self.ASM['hr'])

                        self.logs['recon_psnr'] += self.metrics['recon_psnr'](sr_recon, hr_recon)
                        self.logs['recon_ssim'] += self.metrics['recon_ssim'](sr_recon, hr_recon)

        size = len(self.dataloader)
        for key, value in self.logs.items():
            self.logs[key] = value / size
        
        self.logs['recon_psnr'] /= num_offsets
        self.logs['recon_ssim'] /= num_offsets

        # Print the result of evaluation
        print("***Hologram***")
        print("-------------------------------------------------------------------------------")
        print(f"[Amplitude PSNR] :    {self.logs['amp_psnr']}")
        print(f"[Amplitude SSIM] :    {self.logs['amp_ssim']}")
        print(f"[Phase PSNR] :        {self.logs['phs_psnr']}")
        print(f"[Phase SSIM] :        {self.logs['phs_ssim']}")
        print(f"[Phase Rotate PSNR] : {self.logs['phs_rot_psnr']}")
        print("-------------------------------------------------------------------------------")
        
        print("***Reconstruction***") 
        print("-------------------------------------------------------------------------------")
        print(f"[Recon. PSNR] :  {self.logs['recon_psnr']}")
        print(f"[Recon. SSIM] :  {self.logs['recon_ssim']}")
        print("-------------------------------------------------------------------------------")

    def qualitative(self):
        with torch.no_grad():
            data = self.dataloader.dataset[self.idx]
            lr, hr = self.feed_data(data)
            lr = lr.unsqueeze(0); hr = hr.unsqueeze(0)
            sr = self.inference(lr)

        # inverse-norm of maximum amplitude & phase shift
        shift = True if self.is_inter else False
        sr = self.postprocessing(sr, phs_shift=shift)
        hr = self.postprocessing(hr)
        lr = self.postprocessing(lr)

        # Reconstruction & Save the inference result
        num_offsets = (self.cfg.z[1] - self.cfg.z[0]) / self.cfg.z_offset + 1
        for idx_offset in range(int(num_offsets + 1)):
            z = self.cfg.z[0] + idx_offset * self.cfg.z_offset
            
            lr_recon = opt.reconstruction(lr, z, self.ASM['lr'])
            sr_recon = opt.reconstruction(sr, z, self.ASM['hr'])
            hr_recon = opt.reconstruction(hr, z, self.ASM['hr'])

            save_image(lr_recon, os.path.join(self.cfg.img_path, f'lr_recon_{self.idx:04d}_{idx_offset:02d}_{z:.4f}.png'), normalize=False)
            save_image(sr_recon, os.path.join(self.cfg.img_path, f'sr_recon_{self.idx:04d}_{idx_offset:02d}_{z:.4f}.png'), normalize=False)
            save_image(hr_recon, os.path.join(self.cfg.img_path, f'hr_recon_{self.idx:04d}_{idx_offset:02d}_{z:.4f}.png'), normalize=False)
    
        save_image(lr[:, 0:3], os.path.join(self.cfg.img_path, f'lr_amp_{self.idx:04d}.png'), normalize=False)
        save_image(lr[:, 3:6], os.path.join(self.cfg.img_path, f'lr_phs_{self.idx:04d}.png'), normalize=False)
        save_image(sr[:, 0:3], os.path.join(self.cfg.img_path, f'sr_amp_{self.idx:04d}.png'), normalize=False)
        save_image(sr[:, 3:6], os.path.join(self.cfg.img_path, f'sr_phs_{self.idx:04d}.png'), normalize=False)
        save_image(hr[:, 0:3], os.path.join(self.cfg.img_path, f'hr_amp_{self.idx:04d}.png'), normalize=False)
        save_image(hr[:, 3:6], os.path.join(self.cfg.img_path, f'hr_phs_{self.idx:04d}.png'), normalize=False)

        print("[Info] Qualitative evaluation is done!")

    def run(self):
        # load modules
        self.init_for_module()

        # load dataloader
        self.init_for_dataloader()

        # setup diffraction method
        self.ASM = {
            'lr': opt.AngularSpectrumMethod(
                input_size  = self.cfg.data_size['lr'],
                pixel_pitch = self.cfg.pixelpitch * 2.0,
                wavelength  = self.cfg.wavelength,
                device      = self.dev
            ), 
            'hr': opt.AngularSpectrumMethod(
                input_size  = self.cfg.data_size['hr'],
                pixel_pitch = self.cfg.pixelpitch,
                wavelength  = self.cfg.wavelength,
                device      = self.dev
            )
        }
        
        if self.mode == 0: # quantitative evaluation
            # metrics
            # - the data range of metrics for reconstruction follows tensor holography's baseline
            self.metrics = {
                'amp_psnr'      : PSNR(data_range=1.0).to(self.dev),
                'amp_ssim'      : SSIM(data_range=1.0).to(self.dev),
                'phs_psnr'      : PSNR(data_range=1.0).to(self.dev),
                'phs_ssim'      : SSIM(data_range=1.0).to(self.dev),
                'phs_rot_psnr'  : PhaseRotatePSNR(data_range=1.0).to(self.dev),
                'recon_psnr'    : PSNR(data_range=math.sqrt(2)).to(self.dev),
                'recon_ssim'    : SSIM(data_range=math.sqrt(2)).to(self.dev)
            }

            self.quantitative()
        else:
            # set save path
            if self.is_inter: name = f'{self.cfg.model_name}'
            else: name = f'H2HSR_{self.cfg.model_name}'

            self.cfg.img_path = utl.get_result_path(model_name=name)
            self.qualitative()

