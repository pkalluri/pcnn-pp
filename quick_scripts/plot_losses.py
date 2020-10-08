"Converts .out file into plots of losses during training."

import argparse
import sys
import os
import re
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..'))) # Adds higher directory to python modules path.
from utils import plotting_losses

parser = argparse.ArgumentParser()
parser.add_argument('out_path', type=str, help='Path of the .out file')
parser.add_argument('plot_path', type=str, help='Path to put the plot .png file')
parser.add_argument('-d', '--directory', dest='directory', action='store_true', help='Path is a directory of multiple out files')
# parser.add_argument('-a', '--all_losses', dest='all_losses', action='store_true', help='Should we plot all losses?')
parser.add_argument('-s', '--show', action='store_true', help='Should we show all images now live?')
parser.add_argument('-o', '--overwrite', action='store_true', help='Should we overwrite the existing loss plots?')
# parser.add_argument('-o', '--save_dir', type=str, default='save', help='Out path')
args = parser.parse_args()

if not args.directory: # just one
    plotting_losses.out_to_plots(args.out_path, args.plot_path, show=args.show, overwrite=args.overwrite)
else: # dir of many
    raise('TODO')
#     for filename in os.listdir(args.out_path):
#         if re.match('[^\.].*\.out', filename):
#             filepath = os.path.join(args.out_path, filename)
#             print(f'Trying {filename}')
#             try: # try to get losses
#                 losses = plotting_losses.out_to_losses(filepath, args.all_losses)
#                 _, example_log = list(losses['train'].items())[0]
#                 if example_log: # there is training
#                     plotting_losses.out_to_plots(filepath, args.all_losses, args.show, overwrite=args.overwrite)
#                 else:
#                     raise Exception('No training')
#             except: # dud file or no training, move to archive
#                 # prefix = plotting_losses.out_path2id(filepath, overwrite=args.overwrite)
#                 # new_filepath = os.path.join(args.out_path,'archive', prefix+'.out')
#                 print('Skipping {}'.format(filename))
#                 # print('Skipping {} & moving it to {}'.format(filename, new_filepath))
#                 # os.rename(filepath, new_filepath)
#                 pass # skip this one