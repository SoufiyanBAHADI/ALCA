"""
Created on 30.09.2020
@author: Soufiyan Bahadi
@director: Jean Rouat
@co-director: Eric Plourde
"""

import os
import numpy as np
from scipy.io import wavfile


class DataLoader:
    def __init__(self, context_manager, normalize, batch_size=0):
        self.context_manager = context_manager
        self.normalize = normalize
        self.batch_size = batch_size

    @property
    def context_manager(self):
        return self._context_manager

    @context_manager.setter
    def context_manager(self, value):
        self._context_manager = value

    def __load_heidelburg(self, path=None, language="english", shuffle=True):
        """To use this data loader you need to convert flac files to wav files

        Args:
            path (str, optional): The path of the dataset. Defaults to None.
            language (str, optional): The desired language to load. Defaults to "english".
            shuffle (bool, optional): If True, the loaded dataset will be shuffled. Defaults to True.

        Returns:
            ndarray, ndarray: train set, test set
        """
        np.random.seed(0xbadc0de)
        with open(os.path.join(path, 'train_filenames.txt')) as f:
            train_paths = f.readlines()
        with open(os.path.join(path, 'test_filenames.txt')) as f:
            test_paths = f.readlines()
        max_len = 55718  # is the longest audio signal
        max_len += self.context_manager.stride - (
            max_len -
            self.context_manager.ker_len) % self.context_manager.stride
        train_set = np.zeros((4011, max_len))
        train_idx = np.arange(4011)
        test_set = np.zeros((1079, max_len))
        test_idx = np.arange(1079)
        if shuffle:
            np.random.shuffle(train_idx)
            np.random.shuffle(test_idx)
        i = 0
        for p in train_paths:
            if language in p:
                fs, sig = wavfile.read(
                    os.path.join(path, "audio/" + p.replace('flac\n', 'wav')))
                train_set[train_idx[i], :len(sig)] = sig
                i += 1
        i = 0
        for p in test_paths:
            if language in p:
                fs, sig = wavfile.read(
                    os.path.join(path, "audio/" + p.replace('flac\n', 'wav')))
                test_set[test_idx[i], :len(sig)] = sig
                i += 1
        self.context_manager.fs = fs
        return train_set, test_set

    def load(self, path=None):
        train_set, test_set = self.__load_heidelburg(path)
        
        if self.normalize:
            train_set = train_set / np.linalg.norm(train_set, ord=2,
                                                   axis=1)[:, None]
            test_set = test_set / np.linalg.norm(test_set, ord=2, axis=1)[:,
                                                                          None]
        if self.batch_size > 1:
            train_set = np.array_split(train_set,
                                       len(train_set) // self.batch_size,
                                       axis=0)
            test_set = np.array_split(test_set,
                                      len(test_set) // self.batch_size,
                                      axis=0)
        return train_set, test_set
