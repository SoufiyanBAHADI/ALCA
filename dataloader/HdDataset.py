"""
Created on 30.09.2020
@author: Soufiyan Bahadi
@director: Jean Rouat
@co-director: Eric Plourde
"""

import os
import sys
import numpy as np
from scipy.io import wavfile
from torch.utils.data import Dataset
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from constants import ENGLISH_MAX_LEN, GERMAN_MAX_LEN

class HdDataset(Dataset):
    def __init__(self, context_manager, root_dir, transform, lang="english", eval=False):
        self.transform = transform
        self.__filenames = []
        mode = "test" if eval else "train"
        with open(os.path.join(root_dir, mode+'_filenames.txt')) as f:
            paths = f.readlines()
        for p in paths:
            if lang in p or lang=="both":
                self.__filenames.append(os.path.join(root_dir, "audio/" + p.replace('flac\n', 'wav')))
        if lang=="GERMAN":
            self.max_len = GERMAN_MAX_LEN
        elif lang=="english":
            self.max_len = ENGLISH_MAX_LEN
        else:
            self.max_len = max([ENGLISH_MAX_LEN, GERMAN_MAX_LEN])
        self.max_len += context_manager.stride - (
            self.max_len -
            context_manager.ker_len) % context_manager.stride
    
    def __len__(self):
        return len(self.__filenames)
    
    def __getitem__(self, index):
        _, sig = wavfile.read(os.path.join(self.__filenames[index]))
        rec = np.zeros(self.max_len)
        rec[:len(sig)] = sig
        lab = int(self.__filenames[index][-5])
        sample = {"recording": rec, "label": lab}
        if self.transform:
            sample = self.transform(sample)
        return sample

class ToTensor:
    """Convert data in sample to Tensors."""
    def __call__(self, sample):
        return {"recording": torch.from_numpy(sample['recording'][None, :]).double(),
                "label": torch.Tensor([sample['label']]).long()}

class Normalize:
    """Normalizing signals to a unit energy."""
    def __call__(self, sample):
        rec = sample["recording"]
        return {"recording": rec / torch.linalg.norm(rec, ord=2), "label": sample["label"]}
