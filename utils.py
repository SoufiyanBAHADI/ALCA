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

from constants import CHECKPOINT_CBL, CHECKPOINT_OPT

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device(
    "cpu")


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


def load_optimized_cbl(epoch=None):
    if epoch == 0: # No need to load parameters if the resume epoch is 0
        return None, None, None

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    c = []
    b = []
    filter_ord = []
    path = os.path.join(
        CHECKPOINT_CBL,
        next(os.walk(CHECKPOINT_CBL), (None, None, []))[2][0])
    acc = EventAccumulator(path)
    acc.Reload()
    if epoch is None:
        epoch = len(acc.Scalars('Loss/test')) - 1
    for ci, bi, filter_ordi in zip(acc.Scalars("C/epoch "+str(epoch)), acc.Scalars("b/epoch "+str(epoch)), acc.Scalars("filter_ord/epoch "+str(epoch))):
        c.append([ci[2]])
        b.append([bi[2]])
        filter_ord.append([filter_ordi[2]])
    c = torch.tensor(c, dtype=torch.float64, requires_grad=True, device=device)
    b = torch.tensor(b, dtype=torch.float64, requires_grad=True, device=device)
    filter_ord = torch.tensor(filter_ord,
                              dtype=torch.float64,
                              requires_grad=True,
                              device=device)
    return c, b, filter_ord


def load_optimizer(epoch):
    if epoch == 0:
        return None
    path = os.path.join(
        CHECKPOINT_OPT,
        next(os.walk(CHECKPOINT_OPT), (None, None, []))[2][0])
    return torch.load(path)


def compute_snr(residual, mini_batch):
    res = residual[:, 0].detach().cpu().numpy()
    sig = mini_batch[:, 0].detach().cpu().numpy()
    return np.mean(
        10 * np.log10(np.sum((sig - res)**2, axis=1) / np.sum(res**2, axis=1)))
