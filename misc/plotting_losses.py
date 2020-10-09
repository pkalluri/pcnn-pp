import argparse
import matplotlib.pyplot as plt
import re
import numpy as np
import matplotlib.ticker as ticker
import os
import sys

from misc import training_args as training_args_util
from misc import io as io_util


def save_plot(losses, out_path, title='', separate=True, show=False):
    plt.clf() # clear figure
    max_x = 0
    max_y = 0
    if separate:
        train_losses = losses['train']
        test_losses = losses['test']
        fig, axs = plt.subplots(len(train_losses.keys()), 1)
        for i, loss_name in enumerate(train_losses.keys()):
            ax = axs[i]
            ax.set_title(loss_name)
            ax.plot(list(range(len(train_losses[loss_name]))), train_losses[loss_name], label=f'train')
            ax.plot(list(range(len(train_losses[loss_name]))), test_losses[loss_name], label=f'test')
            ax.legend()
            ax.set_xlim(xmin=0)
            max_y = max(max_y, ax.get_ylim()[1])
            max_x = max(max_x, ax.get_xlim()[1])
    else: # together
        fig, axs = plt.subplots(1, 2)
        for i, subset in enumerate(['train', 'test']):
            ax = axs[i]
            subset_losses = losses[subset]
            for j, (loss_name, loss_log) in enumerate(subset_losses.items()):
                ax.set_title(subset)
                ax.plot(loss_log, label=loss_name)
                max_y = max(max_y, ax.get_ylim()[1])
                max_x = max(max_x, ax.get_xlim()[1])                
            ax.legend()
    plt.suptitle(f'{title}') 
    plt.setp(axs, xlim=(0,max_x), ylim=(0, max_y))
    plt.tight_layout()
    out_path = out_path.split('.png')[0] # strip png extension
    plt.savefig(f'{out_path}.png')
    if show: plt.show()


def out_to_losses(path):
    with open(path, "r") as f:
        txt = f.read()
    losses = {}
    losses['train'] = {}
    losses['test'] = {}

    losses['train']['standard'] = [float(match.split(' = ')[-1]) for match in re.findall('standard_train bits_per_dim = \d+\.\d+', txt)]
    losses['train']['sum'] = [float(match.split(' = ')[-1]) for match in re.findall('sum_train bits_per_dim = \d+\.\d+', txt)]
    losses['train']['max'] = [float(match.split(' = ')[-1]) for match in re.findall('max_train bits_per_dim = \d+\.\d+', txt)]

    losses['test']['standard'] = [float(match.split(' = ')[-1]) for match in re.findall('standard_test bits_per_dim = \d+\.\d+', txt)]
    losses['test']['sum'] = [float(match.split(' = ')[-1]) for match in re.findall('sum_test bits_per_dim = \d+\.\d+', txt)]
    losses['test']['max'] = [float(match.split(' = ')[-1]) for match in re.findall('max_test bits_per_dim = \d+\.\d+', txt)]
    return losses
    

def out_to_plots(out_path, plot_path, show=False, overwrite=False):
    # prefix = out_path2prefix(path)
    # details = out2details(out_path)
    plot_path_base, ext = os.path.splitext(plot_path)
    args = training_args_util.out_to_args(out_path)
    summary_string = training_args_util.args_to_summary_string(args)
    if not os.path.exists(f'{plot_path_base}.pickle') or overwrite:
        losses = out_to_losses(out_path)
        io_util.save_pickle(losses, f'{plot_path_base}.pickle')
        save_plot(losses, f'{plot_path_base}.png', title=summary_string, separate=False, show=show)
        save_plot(losses, f'{plot_path_base}_separate.png', title=summary_string, separate=True, show=show)
    else:
        print(f'File already exists: {plot_path} - Did not overwrite.')
