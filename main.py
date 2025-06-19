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
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import argparse

import torch
from torch.utils.tensorboard import SummaryWriter 
from torch.nn.parallel import DistributedDataParallel as DDP

import configs
from train import * 
from test  import *

if __name__=='__main__':
    """
    Train arguments
        - ep     : the number of epochs
        - model  : model name
            - 0: RDN
            - 1: SwinIR 
            - 2: HAT 
        - resume : resume the training
        - ddp    : training with multi-gpus

    Test arguments
        - eval_mode : evaluation mode
            - 0: Quantitative
            - 1: Qualitative
            - 2: Inference
        - model : model name  
            - 0: RDN
            - 1: SwinIR 
            - 2: HAT 
            - 3: Bicubic (interpolation)
    """
    
    parser = argparse.ArgumentParser(description='H2HSR: Hologram to Hologram Super-Resolution')
    subparser = parser.add_subparsers(dest="mode", help="mode for train or test")

    train_parser = subparser.add_parser("train")
    train_parser.add_argument('--ep',               default=100,    type=int, help='input the number of epochs')
    train_parser.add_argument('--model',            default=0,      type=int, help='input the number of model')
    train_parser.add_argument('--resume',           action='store_true',      help='resume the training')
    train_parser.add_argument('--ddp',              action='store_true',      help='distributed data parallel')
    
    test_parser = subparser.add_parser("test")
    test_parser.add_argument('--eval_mode',         default=0,  type=int, help='quantitative or qualitative')
    test_parser.add_argument('--model',             default=0,  type=int, help='input the number of model')
    test_parser.add_argument('--index',             default=0,  type=int, help='Index on dataset for qualitative')
    args = parser.parse_args()

    if args.mode == 'train':
        config = configs.Config_MIT(args.model, args.ep)
        trainer = H2HSRTrainer(config, args)
        trainer.run()
    else:
        config = configs.Config_MIT(args.model)
        tester = H2HSRTester(config, args)
        tester.run()
        pass
    
    


    

    