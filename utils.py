"""
Created on 30.09.2020
@author: Soufiyan Bahadi
@director: Jean Rouat
@co-director: Eric Plourde
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import butter
from enum import Enum
import os

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device(
    "cpu")


class EnvVar(Enum):
    CHECKPOINT_OPT = "checkpoint/opt"
    CHECKPOINT_CBL = "checkpoint/cbl"


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
    from tensorflow.python.summary.summary_iterator import summary_iterator

    c = []
    b = []
    filter_ord = []
    loss_train = []
    loss_test = []
    mses_train = []
    mses_test = []
    activity_train = []
    activity_test = []

    path = os.path.join(
        EnvVar.CHECKPOINT_CBL.value,
        next(os.walk(EnvVar.CHECKPOINT_CBL.value), (None, None, []))[2][0])
    for e in summary_iterator(path):
        for v in e.summary.value:
            if v.tag == 'Loss/train':
                loss_train.append(v.simple_value)
            elif v.tag == 'Loss/test':
                loss_test.append(v.simple_value)
            elif v.tag == 'MSE/train':
                mses_train.append(v.simple_value)
            elif v.tag == 'MSE/test':
                mses_test.append(v.simple_value)
            elif v.tag == 'Spikes_number/train':
                activity_train.append(v.simple_value)
            elif v.tag == 'Spikes_number/test':
                activity_test.append(v.simple_value)

        if epoch is None:
            epoch = len(loss_train)

        for v in e.summary.value:
            if v.tag == 'C/epoch ' + str(epoch):
                c.append([v.simple_value])
            elif v.tag == 'b/epoch ' + str(epoch):
                b.append([v.simple_value])
            elif v.tag == 'filter_ord/epoch ' + str(epoch):
                filter_ord.append([v.simple_value])
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
        EnvVar.CHECKPOINT_OPT.value,
        next(os.walk(EnvVar.CHECKPOINT_OPT.value), (None, None, []))[2][0])
    return torch.load(path)


def compute_snr(lca):
    res = lca.residual[:, 0].detach().cpu().numpy()
    sig = lca.mini_batch[:, 0].detach().cpu().numpy()
    return np.mean(
        10 * np.log10(np.sum((sig - res)**2, axis=1) / np.sum(res**2, axis=1)))
