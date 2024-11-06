"""
Created on 30.09.2020
@author: Soufiyan Bahadi
@director: Jean Rouat
@co-director: Eric Plourde
"""

import torch
import torch.nn.functional as F

class HardShrink(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, lam):
        return F.hardshrink(u, lam)

    @staticmethod
    def backward(ctx, grad_output):
        # The gradient of this function is set to one
        return grad_output, None
