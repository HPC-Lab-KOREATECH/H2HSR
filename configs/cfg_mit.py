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


from enum import Enum
import math
import utils as ult

"""
    Configuration for MIT-CGH-4K dataset
"""
class Model(Enum):
    RDN = 0
    SwinIR = 1
    HAT = 2
    Bicubic = 3

class Config_MIT:
    def __init__(self, model_num, epochs=0):         
        # info for gpus
        self.num_gpus = 0       # the number of gpus
        
        # dataset
        self.root = './datasets/MIT-CGH-4K/'
        self.dataset = 'MIT-CGH-4K'
        self.data_size = {'lr': 192, 'hr': 384}
        self.batch_size = 16
        self.num_workers = 16

        # training parameters
        self.epochs = epochs
        self.print_freq = 10    # frequency of printing loss during train (iter) 
        self.save_freq = 50     # save model checkpoint (epoch)
        self.learning_rate = 1.0e-4
        self.optimizer = {
            'name': 'Adam',
            'lr': self.learning_rate,
            'beta': (0.99, 0.999)
        }
        self.scheduler = {
            'name': 'StepLR',
            'gamma' : 0.5,
            'step' : self.epochs // 50
        }
        self.max_val = 1.0

        # model properties
        self.model_name = Model(model_num).name

        # hologram parameters
        self.pixelpitch = 8e-6
        self.wavelength = [638e-9, 520e-9, 450e-9] # R,G,B

        # reconstruction
        self.z = [-0.003, 0.003] # distance for reconstruction in validation
        self.z_offset = 0.0001
        self.amp_max = [1.7744140625, 1.744140625, 1.615234375] # R,G,B
        
        # Reconstruction loss parameters
        self.params = {
            'depth_base'        :   -3,
            'depth_scale'       :    6,
            'num_top'           :   15,
            'num_random'        :    5,
            'total_bins'        :  200,
            'attention_weight'  : 0.35,
            
        }               
        
        # checkpoint
        self.save_root = './checkpoints'
        self.save_path = None

        # eval
        self.ckp_root = './checkpoints/trained/'