"""
Created on 30.09.2020
@author: Soufiyan Bahadi
@director: Jean Rouat
@co-director: Eric Plourde
"""

import numpy as np
import torch
import torch.nn.functional as F
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from constants import CHECKPOINT_CBL, CHECKPOINT_OPT

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def reconstruct(kernels, a, stride, flca=False):
    if flca:
        reconstructed_sig = kernels[:, 0, :].T @ a.view((-1, 1))
        reconstructed_sig = reconstructed_sig.view((1, 1, -1))
    else:
        reconstructed_sig = F.conv_transpose1d(a, kernels, stride=stride)
    return reconstructed_sig


def size(tensor):
    s = 1
    for i in tensor.size():
        s = s * i
    return s


def mel_space(low, high, num_channels):
    mel_low = 2595 * np.log10(1 + low / 700)
    mel_high = 2595 * np.log10(1 + high / 700)
    mels = np.linspace(mel_low, mel_high, num=num_channels)
    cf = 700 * (10**(mels / 2595) - 1)
    return cf


def load_optimized_cbl(resume=None):
    if resume == 0: # No need to load parameters if the resume epoch is 0
        return None, None, None
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, CHECKPOINT_CBL)
    path = os.path.join(path, next(os.walk(path), (None, None, []))[2][0])
    acc = EventAccumulator(path)
    acc.Reload()
    epoch = len(acc.Scalars('Loss/test')) - 1 if resume is None else resume - 1
    c = []
    b = []
    filter_ord = []
    for ci, bi, filter_ordi in zip(acc.Scalars("C/epoch "+str(epoch)), acc.Scalars("b/epoch "+str(epoch)), acc.Scalars("filter_ord/epoch "+str(epoch))):
        c.append([ci[2]])
        b.append([bi[2]])
        filter_ord.append([filter_ordi[2]])
    c = torch.tensor(c, dtype=torch.float32, requires_grad=True, device=device)
    b = torch.tensor(b, dtype=torch.float32, requires_grad=True, device=device)
    filter_ord = torch.tensor(filter_ord,
                              dtype=torch.float32,
                              requires_grad=True,
                              device=device)
    return c, b, filter_ord


def load_optimizer(resume):
    if resume == 0:
        return None
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, CHECKPOINT_OPT)
    path = os.path.join(path, next(os.walk(path), (None, None, []))[2][0])
    return torch.load(path, map_location=device)


def compute_snr(reconstructed, signal):
    return 10 * torch.log10(torch.squeeze(reconstructed**2).sum(dim=-1) / (torch.squeeze(signal - reconstructed)**2).sum(dim=-1)).sum().item()
