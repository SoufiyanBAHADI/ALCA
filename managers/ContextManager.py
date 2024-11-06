"""
Created on 30.09.2020
@author: Soufiyan Bahadi
@director: Jean Rouat
@co-director: Eric Plourde
"""

import numpy as np
import torch


class ContextManager:
    def __init__(self,
                 tau,
                 dt,
                 fs,
                 c=None,
                 b=None,
                 filter_ord=None,
                 random_init=False,
                 threshold=6e-3,
                 stride=10,
                 num_channels=32,
                 ker_len=1024,
                 iters=2000,
                 device=torch.device("cuda")):
        self.tau = tau
        self.dt = dt
        self.random_init = random_init
        self.device = device
        self.threshold = threshold
        self.stride = stride
        self.num_channels = num_channels
        self.ker_len = ker_len
        self.iters = iters
        self.device = device
        if c is None:
            self.reset()
        else:
            self.c = c
            self.b = b
            self.filter_ord = filter_ord
        self._t = None
        self._bandwidth = None
        self._central_freq = None
        self.fs = fs

    @property
    def fs(self):
        return self._fs

    @property
    def central_freq(self):
        return self._central_freq

    @property
    def bandwidth(self):
        return self._bandwidth

    @central_freq.setter
    def central_freq(self, value):
        self._central_freq = value
        self._bandwidth = 0.1039 * self._central_freq + 24.7

    @fs.setter
    def fs(self, value):
        if value is not None:
            self._fs = value
            self._t = (torch.arange(
                self.ker_len, dtype=torch.float32, device=self.device).view(
                    1, -1).repeat(self.num_channels, 1) + 1) / self._fs
            if self._central_freq is None:  # If central freq is not set manually
                self._central_freq = torch.from_numpy(self._erb_space()).float().view(
                    -1, 1).to(self.device)
            self._bandwidth = 0.1039 * self._central_freq + 24.7

    def parameters(self):
        # self.central_freq.requires_grad_()
        return [self.c, self.b, self.filter_ord]#, self.central_freq]

    def _erb_space(self, low_freq=100):
        # Glasberg and Moore Parameters
        ear_q = 9.26449
        min_bw = 24.7
        high_freq = self.fs / 2
        # All of the follow_freqing expressions are derived in Apple TR #35, "An
        # Efficient Implementation of the Patterson-Holdsworth Cochlear
        # Filter Bank."  See pages 33-34.
        return -(ear_q * min_bw) + np.exp(
            np.arange(1, self.num_channels + 1) *
            (-np.log(high_freq + ear_q * min_bw) +
             np.log(low_freq + ear_q * min_bw)) /
            self.num_channels) * (high_freq + ear_q * min_bw)

    def compute_weights(self) -> torch.Tensor:
        self._bandwidth = 0.1039 * self._central_freq + 24.7
        weights = self._t**(self.filter_ord - 1) * torch.exp(
            -2 * np.pi * self.b * self._bandwidth *
            self._t) * torch.cos(2 * np.pi * self._central_freq * self._t +
                                  self.c * torch.log(self._t))
        idx = torch.argwhere(torch.norm(weights, p=2, dim=-1) == 0).squeeze()
        weights = (weights / torch.unsqueeze(torch.norm(weights, p=2, dim=-1), dim=-1)).float()
        weights[idx]=0
        return weights

    def reset(self, batch_size = None):
        if batch_size is not None:
            requires_grad = self.c.requires_grad
            with torch.no_grad():
                self.c = torch.stack([self.c]*batch_size)
                self.b = torch.stack([self.b]*batch_size)
                self.filter_ord = torch.stack([self.filter_ord]*batch_size)
            self.c.requires_grad_(requires_grad)
            self.b.requires_grad_(requires_grad)
            self.filter_ord.requires_grad_(requires_grad)

        else:
            if self.random_init:
                self.c = torch.empty((self.num_channels, 1),
                                dtype=torch.float32,
                                requires_grad=True,
                                device=self.device)
                torch.nn.init.xavier_normal_(self.c, 2)
            else:
                self.c = torch.zeros((self.num_channels, 1),
                                dtype=torch.float32,
                                requires_grad=True,
                                device=self.device)
            if self.random_init:
                self.b = torch.empty((self.num_channels, 1),
                    dtype=torch.float32,
                    requires_grad=True,
                    device=self.device)
                torch.nn.init.xavier_normal_(self.c, 2)
            else:
                self.b = torch.tensor([[1]] * self.num_channels,
                                dtype=torch.float32,
                                requires_grad=True,
                                device=self.device)
            if self.random_init:
                self.filter_ord = torch.empty((self.num_channels, 1),
                    dtype=torch.float32,
                    requires_grad=True,
                    device=self.device)
                torch.nn.init.xavier_normal_(self.c, 2)
            else:
                self.filter_ord = torch.tensor([[4]] * self.num_channels,
                                        dtype=torch.float32,
                                        requires_grad=True,
                                        device=self.device)
