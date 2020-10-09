"""
Given a Tensorflow generative model, generate samples from it
"""

import os
import json
import argparse
import re

import numpy as np
import tensorflow as tf

import glob
import shutil

from misc.sampling import sample
from misc.training_args import out_to_updated_args
from misc.restoring_model import restore_model

# TODO split up saved batches into multiple files to avoid OOM, maybe

def get_args():
    parser = argparse.ArgumentParser()
    # The underscores are added so that none of the args will be overwritten by the args from the .out file when we do the merge
    # Frequnetly changed parameters
    parser.add_argument('-pd', '--params_dir_', required=True, type=str,
                        help='The directory where the checkpoint parameters live. Example: save/pretrained')
    parser.add_argument('-tag', '--tag_', type=str, default=None, help='Tag')
    parser.add_argument('-v', '--sampling_variance_', type=float, default=None, help='Sampling variance - we typically either leave out (0) or set to 0.2')

    # Defaulted parameters
    parser.add_argument('-a', '--args_path_', type=str, default=None,
                        help='The .out file to parse arguments from for graph generation')
    parser.add_argument('-cp', '--checkpoint_prefix_', type=str, default=None,
                        help='Checkpoint files prefix - ex. params_cifar.ckpt')
    parser.add_argument('-n', '--num_batches_', type=int,
                        default=500, help='How many batches of samples to generate')
    parser.add_argument('-b', '--batch_size_', type=int,
                        default=100, help='Batch size for generation')
    parser.add_argument('-tdk', '--train_data_key_', type=str,
                        default='samples_arr', help='Key of string for loading npz data')
    parser.add_argument('-si', '--save_interval_', type=int, default=10000,
                        help='How often to save a snapshot of our samples. We always save our final samples at the end.')
    parser.add_argument('-g', '--nr_gpu_', type=int, default=1,
                        help='How many GPUs to distribute across?')

    args = parser.parse_args()

    if args.args_path_:
        args_path = args.args_path_
    else:
        path_parts = re.findall('[^//]+', args.params_dir_)
        path_parts[-1] = path_parts[-1].split('_')[0] + '.out' 
        args_path = ('/').join(path_parts)
        print(args_path)

    args = out_to_updated_args(args_path, args)
    args.sampling_variance = args.sampling_variance_ # should not default to training value


    print('input args:\n', json.dumps(vars(args), indent=4,
          separators=(',', ':')))  # pretty print args
    return args


if __name__ == "__main__":
    args = get_args()
    prefix = args.params_dir_.rstrip('/')
    dest_dir = f'{prefix}_{args.tag_}_samples'
    os.makedirs(dest_dir, exist_ok=True)

    for params_path in glob.glob(args.params_dir_ + '/*.ckpt.*'):
        if not os.path.exists(os.path.join(dest_dir, params_path)):
            print('copying {} to {}'.format(params_path, dest_dir))
            shutil.copy(params_path, dest_dir)

    if args.checkpoint_prefix_:
        ckpt_prefix = args.checkpoint_prefix_
    else:
        ckpt_prefix = 'params.ckpt'

    ckpt_path = os.path.join(args.params_dir_, ckpt_prefix)

    # Uncomment this to include handling of restarting existing sampling process (e.g after a crash)
    # if os.path.exists(latest_samples_path):
    #     print('loading samples from {}'.format(latest_samples_path))
    #     samples_arr = np.load(latest_samples_path)[args.train_data_key_]
    #     samples_list = list(samples_arr)
    #     num_samples_loaded = len(samples_list)
    #     print('loaded {} samples'.format(num_samples_loaded))

    saver, obs_shape, new_x_gen, xs = restore_model(args, args.batch_size_) # note this is where args.variance is passed to the model
    with tf.Session() as sess:
        print('restoring parameters from {} ...'.format(ckpt_path))
        saver.restore(sess, ckpt_path)

        print('generating {} batches of {} samples...'.format(args.num_batches_, args.batch_size_))
        batches = []
        n_samples_generated = 0
        last_save = None
        next_save = args.save_interval_ # next save will happen when this many samples have been generated in total
        for i in range(args.num_batches_):
            print('generating batch {}...'.format(i))
            batches.append(sample(sess, obs_shape, new_x_gen, xs, args.batch_size_, args.nr_gpu_))
            n_samples_generated += args.batch_size_

            if n_samples_generated >= next_save:
                # time to save a checkpoint
                samples = np.concatenate(batches, axis=0)
                samples_path = os.path.join(dest_dir, f'{n_samples_generated}samples.npy')
                print(f'saving {n_samples_generated} samples to {samples_path} ...')
                np.save(samples_path, samples)
                last_save = n_samples_generated
                next_save += args.save_interval_

    if n_samples_generated != last_save:
        print('generating final file')
        samples=np.concatenate(batches, axis=0)
        samples_path=os.path.join(dest_dir, f'{n_samples_generated}samples.npy')
        print(f'saving {n_samples_generated} samples to {samples_path}')
        np.save(samples_path, samples)