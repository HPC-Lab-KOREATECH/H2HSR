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
import cv2
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


"""
    Preprocessing for MIT-CGH-4K dataset 
"""

class MITdataset(Dataset):
    """
        MIT-CGH-4K dataset
        - root: str : path of the dataset
        - mode: str : train or test
        - return: dataset : MIT-CGH-4K dataset
    """
    def __init__(self, 
                 root,
                 mode='train'):

        super().__init__()
        
        self.offset = 0 # the index of dataset 
        self.mode = mode
        # max value of amplitude on the total of holograms
        self.amp_max = torch.FloatTensor([1.7744140625, 1.744140625, 1.615234375])
        
        # low-resolution directory & high-resolution directory
        self.lr_dir = [
            os.path.join(root, '192', 'amplitude', mode), 
            os.path.join(root, '192', 'phase', mode)
        ]

        self.hr_dir = [
            os.path.join(root, '384', 'amplitude', mode),
            os.path.join(root, '384', 'phase', mode)
        ]
        
        self.depth_dir = os.path.join(root, '384', 'depth', mode)
        
        # set the index of the dataset
        if mode == 'train': self.offset = 0
        elif mode == 'valid': self.offset = 3800
        else: self.offset = 3900
   
    def __len__(self):
        if self.mode == 'train':
            return 3800
        else:
            return 100    
    
    def __getitem__(self, idx):
        """
            load 3 channel hologram 
        """
        idx = self.offset + idx
        
        # load the data with opencv       
        lr = torch.cat([
            self.read_exr_to_tensor(self.lr_dir[0], idx),
            self.read_exr_to_tensor(self.lr_dir[1], idx)
        ], dim=2).permute(2, 0, 1) # (H, W, C) -> (C, H, W)

        hr = torch.cat([
            self.read_exr_to_tensor(self.hr_dir[0], idx),
            self.read_exr_to_tensor(self.hr_dir[1], idx)
        ], dim=2).permute(2, 0, 1)

        # load the depth data -> using in reconstruction loss
        depth = self.read_exr_to_tensor(self.depth_dir, idx)[:,:,0]

        # amplitude normalization
        lr[0:3] /= self.amp_max.view(3, 1, 1)
        hr[0:3] /= self.amp_max.view(3, 1, 1)
        return {"lr" : lr, "hr" : hr, "depth" : depth}
    
    def read_exr_to_tensor(self, path, idx):
        """
            read exr file to tensor

            - path: str : path of the exr file
            - idx: int : index of the exr file
            - return: tensor : tensor of the exr file
        """
        data = cv2.cvtColor(cv2.imread(os.path.join(path, '{0:04d}.exr'.format(idx)), 
                            cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        return torch.FloatTensor(data)

