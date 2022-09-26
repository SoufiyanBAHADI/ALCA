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

        if c is None:
            if random_init:
                c = torch.empty((num_channels, 1),
                                dtype=torch.float64,
                                requires_grad=True,
                                device=device)
                torch.nn.init.xavier_normal_(c, 2)
            else:
                c = torch.zeros((num_channels, 1),
                                dtype=torch.float64,
                                requires_grad=True,
                                device=device)
        if b is None:
            if random_init:
                b = torch.empty((num_channels, 1),
                    dtype=torch.float64,
                    requires_grad=True,
                    device=device)
                torch.nn.init.xavier_normal_(c, 2)
            else:
                b = torch.tensor([[1]] * num_channels,
                                dtype=torch.float64,
                                requires_grad=True,
                                device=device)
        if filter_ord is None:
            if random_init:
                filter_ord = torch.empty((num_channels, 1),
                    dtype=torch.float64,
                    requires_grad=True,
                    device=device)
                torch.nn.init.xavier_normal_(c, 2)
            else:
                filter_ord = torch.tensor([[4]] * num_channels,
                                        dtype=torch.float64,
                                        requires_grad=True,
                                        device=device)
        self.c = c
        self.b = b
        self.filter_ord = filter_ord
        self.threshold = threshold
        self.stride = stride
        self.num_channels = num_channels
        self.ker_len = ker_len
        self.iters = iters
        self.device = device
        self.__t = None
        self.__bandwidth = None
        self.__central_freq = None
        self.fs = fs

    @property
    def fs(self):
        return self.__fs
    
    @property
    def central_freq(self):
        return self.__central_freq

    @fs.setter
    def fs(self, value):
        if value is not None:
            self.__fs = value
            self.__t = (torch.arange(
                self.ker_len, dtype=torch.float64, device=self.device).view(
                    1, -1).repeat(self.num_channels, 1) + 1) / self.__fs
            self.__central_freq = torch.from_numpy(self.__erb_space()).view(
                -1, 1).to(self.device)
            self.__bandwidth = 0.1039 * self.__central_freq + 24.7

    def parameters(self):
        return [self.c, self.b, self.filter_ord]

    def __erb_space(self, low_freq=100):
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

    def compute_weights(self):
        weights = self.__t**(self.filter_ord - 1) * torch.exp(
            -2 * np.pi * self.b * self.__bandwidth *
            self.__t) * torch.cos(2 * np.pi * self.__central_freq * self.__t +
                                  self.c * torch.log(self.__t))

        weights = weights / torch.norm(weights, p=2, dim=1).view(-1, 1)
        weights = weights.view(self.__t.shape[0], 1, -1).to(self.device)
        return weights.double()
