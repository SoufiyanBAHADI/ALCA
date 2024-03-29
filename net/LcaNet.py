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
    def __init__(self,
                 cm: ContextManager,
                 lm: LearningManager,
                 pm: PlottingManager = None):
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

    def _criterion(self, mini_batch, activation):
        mse = torch.nn.MSELoss(reduction='none')
        recons = reconstruct(self.weights, activation, self.cm.stride)
        err = 1 / 2 * torch.sum(mse(recons, mini_batch), dim=2)[:, 0]
        return err, recons

    def train(self, mode: bool = True):
        super().train(mode)
        self.cm.c.requires_grad_(mode)
        self.cm.b.requires_grad_(mode)
        self.cm.filter_ord.requires_grad_(mode)

    def eval(self):
        self.train(False)

    def forward(self, mini_batch):
        learning = self.cm.c.requires_grad  # mode train or eval
        sparsity = Sparsity.apply
        shrink = HardShrink.apply
        self.weights = self.cm.compute_weights()
        self.num_shifts = (mini_batch.shape[-1] -
                      self.weights.shape[-1]) // self.cm.stride + 1

        if learning:
            # init states
            # intern potential
            init_u = torch.zeros(mini_batch.shape[0],
                                 self.cm.num_channels,
                                 self.num_shifts,
                                 dtype=torch.float64,
                                 requires_grad=True,
                                 device=self.cm.device)
            # activation
            init_a = torch.zeros(mini_batch.shape[0],
                                 self.cm.num_channels,
                                 self.num_shifts,
                                 dtype=torch.float64,
                                 requires_grad=True,
                                 device=self.cm.device)

            # Hidden states
            # intern potentials
            state_u = [
                init_u - self.cm.dt / self.cm.tau *
                (-F.conv1d(
                    mini_batch, self.weights, stride=self.cm.stride) +
                 init_u + F.conv1d(F.conv_transpose1d(
                     init_a, self.weights, stride=self.cm.stride),
                                   self.weights,
                                   stride=self.cm.stride) - init_a)
            ]
            # activations
            state_a = [shrink(state_u[-1], self.cm.threshold)]

            del init_u
            del init_a

            loss_, _ = self._criterion(mini_batch, state_a[-1])
            loss_ = torch.mean(
                self.lm.alpha * loss_ + self.lm.beta *
                sparsity(state_u[-1], state_a[-1], self.cm.threshold))
        else:
            # if learning is not activated there is no need to define
            # a buffer for hidden states
            u = torch.zeros(mini_batch.shape[0],
                            self.cm.num_channels,
                            self.num_shifts,
                            requires_grad=True,
                            dtype=torch.float64,
                            device=self.cm.device)

        for it in range(self.cm.iters):
            if learning:
                if len(state_u) < self.lm.buffer_size:
                    # Dynamics
                    state_u.append(
                        state_u[-1] + self.cm.dt / self.cm.tau *
                        (F.conv1d(mini_batch,
                                  self.weights,
                                  stride=self.cm.stride) - state_u[-1] -
                         F.conv1d(F.conv_transpose1d(
                             state_a[-1], self.weights, stride=self.cm.stride),
                                  self.weights,
                                  stride=self.cm.stride) + state_a[-1]))

                    # Activation
                    state_a.append(shrink(state_u[-1], self.cm.threshold))
                else:
                    loss_.backward(retain_graph=True)
                    # Optimize c, b, filter_order
                    self.lm.optimizer.step()
                    self.lm.optimizer.zero_grad()
                    # recompute weights
                    with torch.no_grad():
                        self.weights.data = self.cm.compute_weights().data
                    # reset Loss
                    loss_ = 0
                    # last states are new init states
                    init_u = state_u[-1].detach().clone()
                    init_a = state_a[-1].detach().clone()
                    # clear the memory of all states whose loss was backpropagated
                    state_u.clear()
                    state_a.clear()
                    # compute first hidden states
                    state_u.append(
                        init_u + self.cm.dt / self.cm.tau *
                        (F.conv1d(mini_batch,
                                  self.weights,
                                  stride=self.cm.stride) - init_u -
                         F.conv1d(F.conv_transpose1d(
                             init_a, self.weights, stride=self.cm.stride),
                                  self.weights,
                                  stride=self.cm.stride) + init_a))
                    state_a.append(shrink(state_u[-1], self.cm.threshold))
                    del init_a
                    del init_u
                # Loss computation
                mse, recons = self._criterion(mini_batch, state_a[-1])
                sp_err = sparsity(state_u[-1], state_a[-1], self.cm.threshold)

                # Accumulate loss
                loss_ += torch.mean(self.lm.alpha * mse +
                                    self.lm.beta * sp_err)
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
        if learning:
            a = state_a[-1]

        # Save residual
        self.residual = mini_batch - recons

        # Save spikegram
        self.spikegram = a.detach().cpu().numpy().reshape(
            (mini_batch.shape[0], self.cm.num_channels, -1))

        # Save loss at the end of lca
        mse = mse.detach().cpu().numpy()
        sp_err = sp_err.detach().cpu().numpy()
        self.loss = mse + sp_err
        self.mse = 2 * mse / mini_batch.shape[-1]  # torch.mean(self.residual[:, 0].detach()**2, dim=1).cpu().numpy()  # residual energy divided by its dimension
        self.sp_nb = np.linalg.norm(self.spikegram.reshape(
            (self.spikegram.shape[0], -1)),
                                    ord=0,
                                    axis=1)
        if self.pm is not None:
            if not self.pm.track:
                self.pm.append(self.mse, self.sp_nb)
