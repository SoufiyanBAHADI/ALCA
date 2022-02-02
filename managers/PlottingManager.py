"""
Created on 30.09.2020
@author: Soufiyan Bahadi
@director: Jean Rouat
@co-director: Eric Plourde
"""

import matplotlib.pyplot as plt
import os
import numpy as np
from utils import size
from constants import FONT_SIZE, FilterBank


class PlottingManager():
    def __init__(self):
        self.fb = None
        self.dirname = os.path.dirname(__file__)

        self.mses = [[], [], []]
        self.sps = [[], [], []]

        self.__iters = None
        self.mse_track = None
        self.sp_nb_track = None
        self.track = False

    @property
    def iters(self):
        return self.__iters

    @iters.setter
    def iters(self, value):
        if self.__iters is None:
            self.__iters = value
            self.mse_track = np.zeros((len(list(FilterBank)), self.__iters))
            self.sp_nb_track = np.zeros((len(list(FilterBank)), self.__iters))


    def append(self, mse, sp):
        for m, s in zip(mse, sp):
            self.mses[self.fb].append(m.item())
            self.sps[self.fb].append(s.item())

    def track_loss(self, mse, sp, it):
        self.mse_track[self.fb, it] = mse.item()
        self.sp_nb_track[self.fb, it] = size(sp[sp != 0])

    def plot_boxes(self):
        f = plt.figure(figsize=(7, 3))
        plt.rc("axes", axisbelow=True, grid=True)
        plt.rcParams['font.size'] = FONT_SIZE
        plt.boxplot(self.mses, showfliers=False)
        plt.xticks([1, 2, 3], [fb.name for fb in FilterBank])
        plt.ylabel("MSE")
        plt.tight_layout()
        plt.savefig(os.path.join(self.dirname, "../figures/mses.eps"),
                    format="eps")
        
        f = plt.figure(figsize=(7, 3))
        plt.rc("axes", axisbelow=True, grid=True)
        plt.rcParams['font.size'] = FONT_SIZE
        plt.boxplot(self.sps, showfliers=False)
        plt.xticks([1, 2, 3], [fb.name for fb in FilterBank])
        plt.ylabel("Sparsity")
        plt.tight_layout()
        plt.savefig(os.path.join(self.dirname, "../figures/sps.eps"),
                    format="eps")
        

    def plot_loss(self):
        f = plt.figure(figsize=(7, 2.75))

        plt.rc("axes", axisbelow=True, grid=True)
        plt.rcParams['font.size'] = FONT_SIZE
        min_it = np.argmin(
            np.abs(self.mse_track[FilterBank.aGC.value] -
                   self.mse_track[FilterBank.GT.value][-1]))
        its = self.mse_track.shape[-1]

        plt.plot(self.mse_track[FilterBank.aGC.value], label="aGC")
        plt.plot(self.mse_track[FilterBank.GT.value], label="GT")
        plt.hlines(self.mse_track[FilterBank.GT.value][-1],
                   xmin=min_it,
                   xmax=its + 1,
                   linestyles='--',
                   colors='red')
        plt.vlines(min_it,
                   ymin=0,
                   ymax=self.mse_track[FilterBank.aGC.value][min_it],
                   linestyles='--',
                   colors='red')
        plt.xlabel("Iterations")
        plt.xticks(
            np.sort(np.array(list([0, 500, 1000, 1500, 2000] + [min_it]))))
        plt.ylabel("MSE")
        plt.xlim([-10, its + 1])
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.dirname, "../figures/mse_track.eps"),
                    format="eps")
        

        f = plt.figure(figsize=(7, 2.75))

        plt.rc("axes", axisbelow=True, grid=True)
        plt.rcParams['font.size'] = FONT_SIZE
        plt.plot(self.sp_nb_track[FilterBank.aGC.value], label="aGC")
        plt.plot(self.sp_nb_track[FilterBank.GT.value], label="GT")
        plt.vlines(min_it,
                   ymin=0,
                   ymax=self.sp_nb_track[FilterBank.aGC.value, min_it],
                   linestyles='--',
                   colors='red')
        plt.hlines(self.sp_nb_track[FilterBank.aGC.value, min_it],
                   xmin=0,
                   xmax=min_it,
                   linestyles='--',
                   colors='red')
        plt.xlabel("Iterations")
        plt.ylabel("Number of active neurons")
        plt.xticks(
            np.sort(np.array(list([0, 500, 1000, 1500, 2000] + [min_it]))))
        plt.yticks(
            np.sort(
                np.array(
                    list([0, 2000, 4000] +
                         [self.sp_nb_track[FilterBank.aGC.value, min_it]]))))
        plt.xlim([-10, its])
        plt.ylim([0, 4000])

        plt.tight_layout()
        plt.savefig(os.path.join(self.dirname, "../figures/sp_nb_track.eps"),
                    format="eps")
        

    def plot_r(self, weights_list):
        aGC_weights = weights_list[0].detach().cpu().numpy()[::-1, 0, :]
        aGC_R = aGC_weights @ aGC_weights.T
        aGC_R = aGC_R - np.eye(aGC_weights.shape[0])


        GT_weights = weights_list[2].detach().cpu().numpy()[::-1, 0, :]
        GT_R = GT_weights @ GT_weights.T
        GT_R = GT_R - np.eye(GT_weights.shape[0])

        mi = min(np.min(aGC_R), np.min(GT_R))
        ma = min(np.max(aGC_R), np.max(GT_R))
        lim = max(abs(mi), abs(ma))

        f = plt.figure(figsize=(7, 6))

        plt.subplot(121)
        plt.rcParams['font.size'] = FONT_SIZE
        plt.imshow(aGC_R, cmap=plt.cm.RdBu)
        plt.clim(-lim, lim)
        plt.yticks([0, 5, 10, 15])
        plt.xlabel('Channel id')
        plt.ylabel('Channel id')
        plt.grid(False)

        plt.subplot(122)
        plt.rcParams['font.size'] = FONT_SIZE
        plt.imshow(GT_R, cmap=plt.cm.RdBu)
        plt.clim(-lim, lim)
        plt.yticks([0, 5, 10, 15])
        plt.xlabel('Channel id')
        plt.ylabel('Channel id')
        plt.grid(False)

        plt.subplots_adjust(bottom=0.3, right=0.9, top=0.9)
        cax = plt.axes([0.15, 0.1, 0.7, 0.05])
        plt.colorbar(cax=cax, orientation="horizontal")

        plt.savefig(os.path.join(self.dirname, "../figures/GT_R.eps"),
                    format="eps")
        

    def plot_ker(self, weights, fs):
        weights = weights.detach().cpu().numpy()[:, 0, :]
        num_channels = weights.shape[0]

        f = plt.figure(figsize=(7, 2.5))
        plt.rc('axes', axisbelow=True, grid=True)
        plt.rcParams['font.size'] = FONT_SIZE
        for i in range(num_channels):
            plt.magnitude_spectrum(weights[i], Fs=fs, scale='dB')
        #    plt.magnitude_spectrum(original_kernels[i], Fs=fs, scale='dB', ls="--")
        plt.xscale('log')
        plt.xlim(right=fs / 2)
        plt.ylim(top=-20, bottom=-100)
        plt.xlabel('Frequency (Hz)')
        # plt.title(FilterBank(self.fb).name+" filterbank")
        plt.tight_layout()
        plt.savefig(os.path.join(self.dirname, "../figures/ker_spec_"+FilterBank(self.fb).name+".eps"),
                    format="eps")
        
    
    def erb_space(self, chan):
        fs = 48000
        low_freq = 100
        high_freq = fs / 2
        N = 16
        # Glasberg and Moore Parameters
        ear_q = 9.26449
        min_bw = 24.7
        # order = 1

        # All of the follow_freqing expressions are derived in Apple TR #35, "An
        # Efficient Implementation of the Patterson-Holdsworth Cochlear
        # Filter Bank."  See pages 33-34.
        return -(ear_q * min_bw) + np.exp(
            (chan + 1) *
            (-np.log(high_freq + ear_q * min_bw) +
            np.log(low_freq + ear_q * min_bw)) / N) * (high_freq + ear_q * min_bw)
    
    def plot_waveform(self, sig, fs):
        f = plt.figure(figsize=(7, 2))
        plt.rc('axes', axisbelow=True, grid=True)
        plt.rcParams['font.size'] = FONT_SIZE
        
        plt.plot(np.arange(sig.size)*1000/fs, sig)
        # plt.title("Waveform: \"eight\"")
        plt.xlim(left=0, right=20000*1000/fs)
        plt.xlabel("time (ms)")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.dirname, "../figures/waveform.eps"),
                    format="eps")
        

    def plot_spg(self, spg, central_freq, num_channels, num_shifts): 
        idy, idx = np.nonzero(spg)
        #central_freq = central_freq.cpu().numpy().reshape(-1)
        central_freq = np.zeros(num_channels)
        lim = 4
        for i in range(num_channels):
            central_freq[i] = self.erb_space(i)

        f = plt.figure(figsize=(7, 3))
        ax = f.subplots(1, 1)
        plt.rcParams['font.size'] = FONT_SIZE
        f.set_tight_layout(True)

        ax.scatter(idx, central_freq[idy], 1)
        ax.set_xlim(left=0, right=num_shifts)
        ax.set_xlabel("Discrete time samples")
        ax.set_ylabel("Central frequencies (Hz)")
        # ax.set_title("Spikegram using "+FilterBank(self.fb).name+": " + str(len(idx)) + " spikes")
        ax.set_xlim(left=0, right=2000)
        ax.set_yscale("log")
        ax.set_ylim(top=central_freq[lim]) #18290.38611942468
        ax.grid(False)
        # axs[1].set_ylim(bottom=0, top=num_channels)
        
        secax = ax.twinx()
        secax.grid(False)
        secax.set_ylabel('Channel id')
        secax.set_yscale("log")
        secax.set_ylim(ax.get_ylim())
        secax.set_yticks(central_freq[lim:])
        secax.set_yticklabels(num_channels-1-np.arange(lim, num_channels))
        secax.minorticks_off()
        #secax.set_ylim(secax.get_ylim()[::-1])

        #plt.tight_layout()
        plt.savefig(os.path.join(self.dirname, "../figures/spg_"+FilterBank(self.fb).name+".eps"),
                    format="eps")
