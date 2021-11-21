"""
Created on 30.09.2020
@author: Soufiyan Bahadi
@director: Jean Rouat
@co-director: Eric Plourde
"""

import numpy as np
from torch import optim
import torch
from torch.utils.tensorboard import SummaryWriter

from dataloader.DataLoader import DataLoader
from managers.ContextManager import ContextManager
from managers.LearningManager import LearningManager
from managers.PlottingManager import PlottingManager
from constants import FilterBank, Example
from net.LcaNet import Lca
from utils import device, compute_snr, load_optimized_cbl, load_optimizer, EnvVar
import argparse


def run(lca, batches, mode='train', track=False):
    if mode == 'eval':
        lca.eval()
    elif mode == 'train':
        lca.train()

    loss = 0
    act = 0
    mse = 0
    snr = 0
    n = len(batches)

    for it in range(n):
        if it == Example.BATCH_ID.value and track: # just for plotting tracked info of the chosen example
            iters = lca.cm.iters
            lca.cm.iters = 2048
            lca.pm.iters = lca.cm.iters
            lca.pm.track = True
            lca.mini_batch = batches[it]
            lca()
            lca.pm.track = False
            lca.cm.iters = iters
            lca.pm.plot_waveform(batches[it][Example.SIG_ID.value], lca.cm.fs)
            lca.pm.plot_spg(lca.spikegram[Example.SIG_ID.value], lca.cm.central_freq, lca.cm.num_channels, lca.num_shifts)
        lca.mini_batch = batches[it]
        lca.lm.optimizer.zero_grad()
        lca()
        act += np.mean(lca.sp_nb)
        mse += np.mean(lca.mse)
        loss += np.mean(lca.loss)
        snr += compute_snr(lca)

    return loss / n, act / n, mse / n, snr / n


def fit(lca, train_set, test_set, eval, plot, start):
    if not eval:
        # tensorboard log
        writer = SummaryWriter(log_dir="checkpoint/cbl")
        for e in range(start, lca.lm.epochs + start, 1):
            print(f'epoch {e}:')
            # Train
            loss, act, mse, snr = run(lca, train_set, mode='train')
            # optimizer checkpoint
            torch.save(lca.lm.optimizer, EnvVar.CHECKPOINT_OPT.value)
            # Write train results
            writer.add_scalar('Loss/train', loss, e)
            writer.add_scalar('Spikes number/train', act, e)
            writer.add_scalar('MSE/train', mse, e)
            writer.add_scalar('SNR/train', snr, e)
            print(
                f'\tTRAIN: loss = {loss:.5f} / snr = {snr:.2f} dB / spike number = {act}'
            )
            # Write learnable params state
            for i in range(lca.cm.num_channels):
                writer.add_scalar('C/epoch ' + str(e),
                                  lca.cm.c[i].item(),
                                  global_step=i)
                writer.add_scalar('b/epoch ' + str(e),
                                  lca.cm.b[i].item(),
                                  global_step=i)
                writer.add_scalar('filter_ord/epoch ' + str(e),
                                  lca.cm.filter_ord[i].item(),
                                  global_step=i)
            # Test
            loss, act, mse, snr = run(lca, test_set, mode='eval')
            # Write test results
            writer.add_scalar('Loss/test', loss, e)
            writer.add_scalar('Spikes number/test', act, e)
            writer.add_scalar('MSE/test', mse, e)
            writer.add_scalar('SNR/test', snr, e)
            print(
                f'\tTEST: loss = {loss:.5f} / snr = {snr:.2f} dB / spike number = {act}'
            )
        writer.close()
    else:
        if plot:
            c_aGC, b_aGC, filter_ord_aGC = load_optimized_cbl(10)
            c_cGC = torch.tensor([[0.979]] * lca.cm.num_channels,
                                 dtype=torch.float64,
                                 requires_grad=False,
                                 device=device)
            b_cGC = torch.tensor([[1.14]] * lca.cm.num_channels,
                                 dtype=torch.float64,
                                 requires_grad=False,
                                 device=device)
            filter_ord_cGC = torch.tensor([[4]] * lca.cm.num_channels,
                                          dtype=torch.float64,
                                          requires_grad=False,
                                          device=device)
            c_GT = torch.tensor([[0]] * lca.cm.num_channels,
                                dtype=torch.float64,
                                requires_grad=False,
                                device=device)
            b_GT = torch.tensor([[1]] * lca.cm.num_channels,
                                dtype=torch.float64,
                                requires_grad=False,
                                device=device)
            filter_ord_GT = torch.tensor([[4]] * lca.cm.num_channels,
                                         dtype=torch.float64,
                                         requires_grad=False,
                                         device=device)
            w = []
            for i in range(3):
                if i == 0:
                    c, b, filter_ord = c_aGC, b_aGC, filter_ord_aGC
                    lca.pm.fb = FilterBank.aGC.value
                elif i == 1:
                    c, b, filter_ord = c_cGC, b_cGC, filter_ord_cGC
                    lca.pm.fb = FilterBank.cGC.value
                else:
                    c, b, filter_ord = c_GT, b_GT, filter_ord_GT
                    lca.pm.fb = FilterBank.GT.value
                
                lca.cm.c, lca.cm.b, lca.cm.filter_ord = c, b, filter_ord
                w.append(lca.cm.compute_weights())
                lca.pm.plot_ker(w[-1], lca.cm.fs)
                loss, act, mse, snr = run(lca, test_set, mode='eval',track=True)
                print(FilterBank(i).name, ":", act)
            lca.pm.plot_r(w)
            lca.pm.plot_boxes()
            lca.pm.plot_loss()
        else:
            # Test
            loss, act, mse, snr = run(lca, test_set, mode='eval')
            print(
                f'TEST: loss = {loss:.5f} / snr = {snr:.2f} dB / spike number = {act}'
            )


def main(args):
    # reproducibility
    torch.manual_seed(0xbadc0de)
    # Create LCA parameters
    c, b, filter_ord = load_optimized_cbl(args.resume)
    optimizer = load_optimizer(args.resume)
    cm = ContextManager(tau=args.tau,
                        dt=args.dt,
                        c=c,
                        b=b,
                        filter_ord=filter_ord,
                        random_init=args.random_init,
                        threshold=args.threshold,
                        stride=args.stride,
                        num_channels=args.num_chan,
                        ker_len=args.ker_len,
                        iters=args.iters,
                        device=device)
    # Load data
    data_loader = DataLoader(cm, normalize=True, batch_size=args.batch_size)
    train_set, test_set = data_loader.load(args.path)
    # Create learning parameters
    if optimizer is None:
        if args.optimizer == 'adam':
            optimizer = optim.Adam(cm.parameters(), lr=args.lr, amsgrad=True)
        elif args.optimizer == 'sgd':
            optimizer = optim.SGD(cm.parameters(), lr=args.lr, amsgrad=True)
        else:
            optimizer = optim.Adam(cm.parameters(), lr=args.lr, amsgrad=True)
    else:
        optimizer.param_groups[0]['params'] = cm.parameters()
    
    # Solved with checkpoint
    # for _ in range(args.resume):
        # for __ in range(len(train_set)):
            # for i in range(64):
                # optimizer.step()
                # if (i + 1) % 8 == 0 and i != 0:
                    # optimizer.zero_grad()
    lm = LearningManager(optimizer=optimizer,
                         buffer_size=args.buffer_size,
                         epochs=args.epochs)
    pm = None
    if args.plot:
        pm = PlottingManager()
    # Create LCA module
    lca = Lca(cm=cm, lm=lm, pm=pm)
    # Fit the module
    fit(lca, train_set, test_set, args.eval, args.plot, args.resume)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='ALCA',
        usage='%(prog)s [options]',
        description='Adaptive aproach applied to LCA for audio.')

    parser.add_argument('-p',
                        '--path',
                        default="/archive/bahs2702/heidelberg",
                        type=str,
                        help='The path of the data set.')
    parser.add_argument('--tau',
                        type=float,
                        default=1e-2,
                        help='Neurons\' time constant.')
    parser.add_argument('--dt',
                        type=float,
                        default=1e-4,
                        help='Euler\'s resolution method clock.')
    parser.add_argument('--threshold',
                        type=float,
                        default=3e-3,
                        help='Firing threshold.')
    parser.add_argument('--stride', type=int, default=10, help='Stride size.')
    parser.add_argument('--ker-len',
                        type=int,
                        default=1024,
                        help='Kernels\' length.')
    parser.add_argument('--num-chan',
                        type=int,
                        default=16,
                        help='Number of channels.')
    parser.add_argument('--iters',
                        type=float,
                        default=64,
                        help='The LCA\'s iterations.')
    parser.add_argument('--optimizer',
                        choices=['sgd', 'adam'],
                        default='adam',
                        type=str,
                        help='The optimizer needed for training.')
    parser.add_argument('--lr',
                        type=float,
                        default=2e-4,
                        help='Learning rate.')
    parser.add_argument('--batch-size',
                        type=int,
                        default=8,
                        help='the size of each mini batch.')
    parser.add_argument(
        '--buffer-size',
        type=int,
        default=8,
        help=
        'The size of the buffer where to store steady states for backpropagation through time algorithm'
    )
    parser.add_argument('-e',
                        '--epochs',
                        type=int,
                        default=20,
                        help='number of epochs.')
    parser.add_argument(
        '--eval',
        action='store_true',
        help=
        'Specifies the evaluation. If false the algorithm will run in training mode'
    )
    parser.add_argument('-v',
                        '--verbose',
                        action='store_true',
                        help='allows the program verbosity')
    parser.add_argument('--random-init',
                        action='store_true',
                        help='parameters are initiallized randomly')
    parser.add_argument('--resume',
                        type=int,
                        default=0,
                        help='The epoch from which the learning will resume')
    parser.add_argument(
        '--plot',
        action='store_true',
        help=
        'If specified the program will plot all outputs. --eval should be specified'
    )
    args = parser.parse_args()
    main(args)
