"""
Created on 30.09.2020
@author: Soufiyan Bahadi
@director: Jean Rouat
@co-director: Eric Plourde
"""

import torch


class Sparsity(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, a, lam):
        ctx.save_for_backward(u, a)
        return lam * (lam/2 * torch.norm(a.view((a.shape[0], -1)), p=0, dim=1)) # C = lam/2 * l0-norm

    @staticmethod
    def backward(ctx, grad_output):
        u, a, = ctx.saved_tensors
        grad_a = u - a
        return grad_a, grad_a, None