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
import torch

import torch.nn.functional as F
from torch import nn
from torch.fft import fftn, fftshift, ifftn, ifftshift

def amp_phs_to_complex(amp, phs): # amp & phs to complex hologram
    phs = (phs - 0.5) * 2. * math.pi
    return torch.complex(amp * torch.cos(phs), amp * torch.sin(phs))

def complex_to_amp_phs(comp): # complex to amp & phs
    return torch.abs(comp), torch.angle(comp)

def reconstruction(x, z, ASM):
    # x: [B, C, H, W]
    # - [:, 0:3] -> amplitude
    # - [:, 3:6] -> phase
    x_pad = ASM.padding(amp_phs_to_complex(x[:, 0:3], x[:, 3:6]))
    recon = ASM.propagation(ASM.FourierTransform(x_pad), z)
    return torch.abs(ASM.unpadding(recon))

def diffract(x, z, ASM):
    # x: [B, C, H, W]
    # - [:, 0:3] -> amplitude
    # - [:, 3:6] -> phase
    x_pad = ASM.padding(amp_phs_to_complex(x[:, 0:3], x[:, 3:6]))
    recon = ASM.unpadding(ASM.propagation(ASM.FourierTransform(x_pad), z))

    amp, phs = complex_to_amp_phs(recon)
    return torch.cat([amp, phs], dim=1)

    
class Propagation(nn.Module):
    def __init__(self,
                 input_size,
                 pixel_pitch,
                 wavelength,
                 device='cpu',
                 dtype=torch.float32
                 ):
        super(Propagation, self).__init__()
    
        # base information
        self.input_size  = None
        self.pixel_pitch = None
        self.wavelength  = None
        self.device      = None
        self.dtype       = None
        self.pad         = None
        self.resolution  = None

        # compute parameter
        self.FX          = None
        self.FY          = None
        self.hexp        = None

    def FourierTransform(self, x):
        return fftshift(fftn(fftshift(x, dim=(-2,-1)), norm='ortho', dim=(-2,-1)), dim=(-2,-1))

    def InverseFourierTransform(self, x):
        return ifftshift(ifftn(ifftshift(x, dim=(-2,-1)), norm='ortho', dim=(-2,-1)), dim=(-2,-1))

    def transferFunction(self):
        pass

class AngularSpectrumMethod(Propagation):
    """
        Angular Spectrum Method (ASM)
        * all physical params use meter (m) unit.
        - input_size: tuple -> input size (W, H)
        - pixel_pitch: float -> pixel pitch (um)
        - wavelength: float list -> wavelength (nm)
        - device: str -> device (cpu or cuda)
        - dtype: torch.dtype -> data type (torch.float32 or torch.float64)
    """
    
    def __init__(self,
                input_size,
                pixel_pitch,
                wavelength,
                device='cpu',
                dtype=torch.float32
                ):
        super(AngularSpectrumMethod, self).__init__(
            input_size, pixel_pitch, wavelength, device, dtype
        )

        # base information
        if not isinstance(input_size, (list, tuple)):
            input_size = [input_size, input_size]

        self.width          = input_size[0]
        self.height         = input_size[1]
        self.pixel_pitch    = pixel_pitch
        self.wavelength     = torch.FloatTensor(wavelength)[None, :, None, None] # C => [B, C, H, W]
        self.dtype          = dtype
        self.device         = device
        self.pad            = [input_size[0] // 2, input_size[1] // 2]
        self.resolution     = [input_size[0] * 2,  input_size[1] * 2]

        # compute parameter
        self.FX             = 0
        self.FY             = 0
        self.hexp           = self.transferFunction().to(device)

    def padding(self, x):
        # adding zero-padding
        padded = F.pad(x, (self.pad[1], self.pad[1], self.pad[0], self.pad[0]), "constant", 0)
        return padded
    
    def unpadding(self, x):
        # unpadding the input
        return x[:, :, self.pad[0]:-self.pad[0], self.pad[1]:-self.pad[1]]

    def transferFunction(self):
        """
            transfer function of ASM: h(x)
        """
        k = 2 * math.pi * torch.reciprocal(self.wavelength)
        dfx = 1 / (self.resolution[1] * self.pixel_pitch)
        dfy = 1 / (self.resolution[0] * self.pixel_pitch)

        fx = torch.linspace(-self.resolution[1] / 2, (self.resolution[1] / 2 -1), self.resolution[1]) * dfx
        fy = torch.linspace(-self.resolution[0] / 2, (self.resolution[0] / 2 -1), self.resolution[0]) * dfy

        self.FX, self.FY = torch.meshgrid(fx, fy, indexing='xy')

        return k * torch.sqrt(1 - (self.wavelength * self.FX)**2 - (self.wavelength * self.FY)**2)

    def propagation(self, x_fft, z_distance):
        hx  = torch.exp(1j * self.hexp * z_distance).to(self.device)
        hx[(1 - (self.wavelength * self.FX)**2 - (self.wavelength * self.FY)**2) < 0]=0

        u2 =self.InverseFourierTransform(x_fft * hx)
        return u2