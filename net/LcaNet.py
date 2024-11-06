"""
Created on 30.09.2020
@author: Soufiyan Bahadi
@director: Jean Rouat
@co-director: Eric Plourde
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from managers.ContextManager import ContextManager
from managers.LearningManager import LearningManager
from managers.PlottingManager import PlottingManager
from net.HardShrink import HardShrink
from net.Sparsity import Sparsity
from utils import reconstruct, size
from constants import Example


class Lca(nn.Module):
    def __init__(self, cm: ContextManager, lm: LearningManager, pm: PlottingManager = None):
        super(Lca, self).__init__()
        # managers init
        self.cm = cm
        self.lm = lm
        self.pm = pm

        # weights
        self.weights = None
        # num_shifts
        self.numshifts = None

        # mini_batch and Residu
        self.residual = None

        # Final LCA outputs
        self.spikegram = None

        # Final LCA error
        self.loss = None
        self.mse = None
        self.sp_nb = None
        self.mse = torch.nn.MSELoss(reduction="none")

    def _criterion(self, mini_batch, activation):
        recons = reconstruct(self.weights, activation, self.cm.stride)
        err = 1 / 2 * torch.sum(self.mse(recons, mini_batch), dim=2)[:, 0]
        return err, recons

    def train(self, mode: bool = True):
        super().train(mode)
        self.cm.c.requires_grad_(mode)
        self.cm.b.requires_grad_(mode)
        self.cm.filter_ord.requires_grad_(mode)


    def eval(self):
        self.train(False)

    def forward(self, mini_batch):
        sparsity = Sparsity.apply
        shrink = HardShrink.apply
        self.weights = self.cm.compute_weights()
        self.num_shifts = (mini_batch.shape[-1] - self.weights.shape[-1]) // self.cm.stride + 1

        if self.training:
            loss_ = torch.zeros(1, device=self.cm.device)
        u = torch.zeros(
            mini_batch.shape[0],
            self.cm.num_channels,
            self.num_shifts,
            dtype=torch.float32,
            requires_grad=True,
            device=self.cm.device,
        )
        a = torch.zeros_like(u)

        for it in range(self.cm.iters):
            if self.training:
                if (it + 1) % self.lm.buffer_size == 0:
                    u = u.clone().detach()
                    a = a.clone().detach()
                u = u + self.cm.dt / self.cm.tau * (
                    F.conv1d(mini_batch, self.weights, stride=self.cm.stride)
                    - u
                    - F.conv1d(F.conv_transpose1d(a, self.weights, stride=self.cm.stride), self.weights, stride=self.cm.stride)
                    + a
                )
                a = shrink(u, self.cm.threshold)

                mse, recons = self._criterion(mini_batch, a)
                sp_err = sparsity(u, a, self.cm.threshold)
                loss_ += torch.mean(mse + self.lm.beta * sp_err)
            else:
                # Activation
                a = shrink(u, self.cm.threshold)

                # Loss computation
                mse, recons = self._criterion(mini_batch, a)
                sp_err = sparsity(u, a, self.cm.threshold)
                loss = mse + sp_err

                # Computing loss gradients
                loss.sum().backward()

                with torch.no_grad():
                    # Dynamics
                    u.data.sub_(u.grad, alpha=self.cm.dt / self.cm.tau)
                    u.grad.zero_()

            if self.pm is not None:
                if self.pm.track:
                    # Tracking data
                    self.pm.track_loss(2 * mse[Example.SIG_ID.value] / mini_batch.shape[-1], a[Example.SIG_ID.value], it)

        # Save residual
        self.residual = mini_batch - recons

        # Save spikegram
        self.spikegram = a.detach().cpu().numpy().reshape((mini_batch.shape[0], self.cm.num_channels, -1))

        # Save loss at the end of lca
        mse = mse.detach().cpu().numpy()
        sp_err = sp_err.detach().cpu().numpy()
        loss = mse + sp_err
        # torch.mean(self.residual[:, 0].detach()**2, dim=1).cpu().numpy()  # residual energy divided by its dimension
        sp_nb = np.linalg.norm(self.spikegram.reshape((self.spikegram.shape[0], -1)), ord=0, axis=1)
        if self.pm is not None:
            if not self.pm.track:
                self.pm.append(self.mse, self.sp_nb)
        if self.training:
            return loss_, loss, mse, sp_nb
        else:
            return loss, mse, sp_nb
