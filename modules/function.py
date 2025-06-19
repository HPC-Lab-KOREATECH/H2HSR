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
import torch.nn as nn
from torch import Tensor, tensor
from typing_extensions import Literal

from functools import partial
from typing import Any, Optional, Sequence, Tuple, Union

from torchmetrics.functional.image.psnr import _psnr_compute, _psnr_update
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn

import optics as opt

# Translated from TensorFlow (original: https://github.com/liangs111/tensor_holography)
# Based on: Shi et al., Nature 2021 — "Toward real-time photorealistic 3D holography with deep neural networks"
class HoloLoss(nn.Module):
    """ 
        The following section is intentionally left blank due to license restrictions.
        In compliance with the Tensor Holography license agreement, this part of the code
        cannot be disclosed or published.
    """

    def __init__(self):
        super().__init__()

    # comparison between two phases on polar coordinates
    def diff(self, p1, p2): 
        pass
    
    def forward(self, sr, hr):
        pass
 
class ASMLoss(nn.Module):
    """ 
        The following section is intentionally left blank due to license restrictions.
        In compliance with the Tensor Holography license agreement, this part of the code
        cannot be disclosed or published.
    """

    def __init__(self, ASM, params):
        super().__init__()    
        
    def forward(self, sr, hr, depth):
        pass
        
    def getDepthFocus(self, depth):
        pass

    def getPerceptualLoss(self, sr, hr, depth, depth_focus):
        pass
    
    def getASM(self, fft_sr, fft_hr, depth_focus): 
        pass

    def getTV(self, sr, hr):
        pass
    
    def getVariation(self, recon):
        pass

    def getDepthWeight(self, depth, depth_focus, depth_scale):
        pass



# This implementation is modified from TorchMetrics' legacy PSNR module:
# https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/image/psnr.py
#
# It is adapted to reflect the method proposed in:
# Oh, J., Kim, H., & Lee, B. (2022). "A new objective quality metric for phase hologram processing."
# ETRI Journal, 44(2), 283–293. https://doi.org/10.4218/etrij.2022-0087
class PhaseRotatePSNR(Metric):
    r"""`Compute Peak Signal-to-Noise Ratio`_ (PSNR).

    .. math:: \text{PSNR}(I, J) = 10 * \log_{10} \left(\frac{\max(I)^2}{\text{MSE}(I, J)}\right)

    Where :math:`\text{MSE}` denotes the `mean-squared-error`_ function.

    As input to ``forward`` and ``update`` the metric accepts the following input

    - ``preds`` (:class:`~torch.Tensor`): Predictions from model of shape ``(N,C,H,W)``
    - ``target`` (:class:`~torch.Tensor`): Ground truth values of shape ``(N,C,H,W)``

    As output of `forward` and `compute` the metric returns the following output

    - ``psnr`` (:class:`~torch.Tensor`): if ``reduction!='none'`` returns float scalar tensor with average PSNR value
      over sample else returns tensor of shape ``(N,)`` with PSNR values per sample

    Args:
        data_range:
            the range of the data. If None, it is determined from the data (max - min). If a tuple is provided then
            the range is calculated as the difference and input is clamped between the values.
            The ``data_range`` must be given when ``dim`` is not None.
        base: a base of a logarithm to use.
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'`` or ``None``: no reduction will be applied

        dim:
            Dimensions to reduce PSNR scores over, provided as either an integer or a list of integers. Default is
            None meaning scores will be reduced across all dimensions and all batches.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError:
            If ``dim`` is not ``None`` and ``data_range`` is not given.

    Example:
        >>> from torchmetrics.image import PeakSignalNoiseRatio
        >>> psnr = PeakSignalNoiseRatio()
        >>> preds = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
        >>> target = torch.tensor([[3.0, 2.0], [1.0, 0.0]])
        >>> psnr(preds, target)
        tensor(2.5527)

    """
    is_differentiable: bool = True
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0

    min_target: Tensor
    max_target: Tensor

    def __init__(
        self,
        data_range: Optional[Union[float, Tuple[float, float]]] = None,
        base: float = 10.0,
        reduction: Literal["elementwise_mean", "sum", "none", None] = "elementwise_mean",
        dim: Optional[Union[int, Tuple[int, ...]]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if dim is None and reduction != "elementwise_mean":
            rank_zero_warn(f"The `reduction={reduction}` will not have any effect when `dim` is None.")

        if dim is None:
            self.add_state("sum_squared_error", default=tensor(0.0), dist_reduce_fx="sum")
            self.add_state("total", default=tensor(0), dist_reduce_fx="sum")
        else:
            self.add_state("sum_squared_error", default=[], dist_reduce_fx="cat")
            self.add_state("total", default=[], dist_reduce_fx="cat")

        self.clamping_fn = None
        if data_range is None:
            if dim is not None:
                # Maybe we could use `torch.amax(target, dim=dim) - torch.amin(target, dim=dim)` in PyTorch 1.7 to
                # calculate `data_range` in the future.
                raise ValueError("The `data_range` must be given when `dim` is not None.")

            self.data_range = None
            self.add_state("min_target", default=tensor(0.0), dist_reduce_fx=torch.min)
            self.add_state("max_target", default=tensor(0.0), dist_reduce_fx=torch.max)
        elif isinstance(data_range, tuple):
            self.add_state("data_range", default=tensor(data_range[1] - data_range[0]), dist_reduce_fx="mean")
            self.clamping_fn = partial(torch.clamp, min=data_range[0], max=data_range[1])
        else:
            self.add_state("data_range", default=tensor(float(data_range)), dist_reduce_fx="mean")
        self.base = base
        self.reduction = reduction
        self.dim = tuple(dim) if isinstance(dim, Sequence) else dim

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        if self.clamping_fn is not None:
            preds = self.clamping_fn(preds)
            target = self.clamping_fn(target)

        # Modified part to calculate the accuracy of phase to consider the propoerty of rotation [0, 1] 
        diff = torch.abs(preds - target)
        preds = torch.where(diff > 0.5, 1.0 - diff + target, preds)

        sum_squared_error, n_obs = _psnr_update(preds, target, dim=self.dim)      
        
        if self.dim is None:
            if self.data_range is None:
                # keep track of min and max target values
                self.min_target = torch.minimum(target.min(), self.min_target)
                self.max_target = torch.maximum(target.max(), self.max_target)

            self.sum_squared_error += sum_squared_error
            self.total += n_obs
        else:
            self.sum_squared_error.append(sum_squared_error)
            self.total.append(n_obs)

    def compute(self) -> Tensor:
        """Compute peak signal-to-noise ratio over state."""
        data_range = self.data_range if self.data_range is not None else self.max_target - self.min_target

        if self.dim is None:
            sum_squared_error = self.sum_squared_error
            total = self.total
        else:
            sum_squared_error = torch.cat([values.flatten() for values in self.sum_squared_error])
            total = torch.cat([values.flatten() for values in self.total])
        return _psnr_compute(sum_squared_error, total, data_range, base=self.base, reduction=self.reduction)