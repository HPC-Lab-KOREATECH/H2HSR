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


from torch import nn
from torch import optim
from torch.optim import lr_scheduler

def load_optimizer(model, args):
    optimizer = None
    if args['name'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), args['lr'], args['beta'])
    else:
        raise ValueErorr(f"[Error] {args['name']} is not implement yet.")
    return optimizer    

def load_scheduler(optimizer, args):
    scheduler = None
    if args['name'] == 'StepLR':
        scheduler = lr_scheduler.StepLR(optimizer, args['step'], args['gamma'])
    else:
        raise ValueErorr(f"[Error] {args['name']} is not implement yet.")
    return scheduler
    