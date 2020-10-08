"""
Given a Tensorflow generative model, generate samples from it
"""

import os
import json
import argparse

import numpy as np
import tensorflow as tf

import glob
import shutil

from utils.sample_from_model import sample_from_model
from utils.args_reading import overwrite_args_from_out_file
from utils.recreate_model import recreate_model

# TODO split up saved batches into multiple files to avoid OOM, maybe


def get_args():
    parser = argparse.ArgumentParser()
    # The underscores are added so that none of the args will be overwritten by the args from the .out file

    # Oft-changed parameters
    parser.add_argument('-pd', '--params_dir_', required=True, type=str, help='The directory where the checkpoint parameters live. Example: save/pretrained')
    parser.add_argument('-v', '--sampling_variance', type=float, default=None, help='Sampling variance - we typically either leave out (0) or set to 0.2')

    # Defaulted parameters
    parser.add_argument('-a', '--args_file_', type=str, default=None, help='The .out file to parse arguments from for graph generation')
    parser.add_argument('-cp', '--checkpoint_prefix_', type=str, default=None, help='Checkpoint files prefix - ex. cifar_params.cpkt')
    parser.add_argument('-nb', '--num_batches_', type=int, default=5000, help='How many batches of samples to generate')
    parser.add_argument('-b', '--batch_size_', type=int, default=10, help='Batch size for generation')
    parser.add_argument('-tdk', '--train_data_key_', type=str, default='samples_np', help='Key of string for loading npz data')
    parser.add_argument('-si', '--save_interval_', type=int, default=100, help='How often to save a snapshot of our samples')
    parser.add_argument('-g', '--nr_gpu_', type=int, default=1, help='How many GPUs to distribute the training across?')

    args = parser.parse_args()

    if args.args_file_:
        args_file = args.args_file_
    else:
        args_file = args.params_dir_ + '.out'

    args = overwrite_args_from_out_file(args_file, args)

    print('input args:\n', json.dumps(vars(args), indent=4, separators=(',',':'))) # pretty print args
    return args


if __name__ == "__main__":
    args = get_args()

    sampling_variance_str = 'var0'
    if args.sampling_variance:
        sampling_variance_str = 'var'+str(args.sampling_variance).replace('.', '_')

    out_dir = args.params_dir_ + '_samples_' + sampling_variance_str
    os.makedirs(out_dir, exist_ok=True)

    for file in glob.glob(args.params_dir_ + '/*.ckpt.*'):
        if not os.path.exists(os.path.join(out_dir, file)):
            print('copying {} to {}'.format(file, out_dir))
            shutil.copy(file, out_dir)

    if args.checkpoint_prefix_:
        ckpt_prefix = args.checkpoint_prefix_
    else:
        ckpt_prefix = 'params_' + args.data_set + '.ckpt'

    ckpt_path = os.path.join(out_dir, ckpt_prefix)
    latest_samples_path = os.path.join(out_dir, 'samples.latest.npz')

    if os.path.exists(latest_samples_path):
        print('loading samples from {}'.format(latest_samples_path))
        samples_np = np.load(latest_samples_path)[args.train_data_key_]
        samples_list = list(samples_np)
        num_samples_loaded = len(samples_list)
        print('loaded {} samples'.format(num_samples_loaded))

    saver, obs_shape, new_x_gen, xs = recreate_model(args, args.batch_size_)
    with tf.Session() as sess:
        print('restoring parameters from {} ...'.format(ckpt_path))
        saver.restore(sess, ckpt_path)

        print('generating {} batches of {} samples...'.format(args.num_batches_, args.batch_size_))
        samples = []
        save_interval_counter = 1
        for i in range(args.num_batches_):
            print('generating batch {}...'.format(i))
            samples.append(sample_from_model(sess, obs_shape, new_x_gen, xs, args.batch_size_, args.nr_gpu_))

            samples_np = np.concatenate(samples,axis=0)
            samples_len = len(list(samples_np))

            print('saving {} samples to {} ...'.format(samples_len, latest_samples_path))
            np.savez(latest_samples_path, samples_np=samples_np)

            if samples_len >= (save_interval_counter * args.save_interval_):
                intermediate_samples_path = os.path.join(out_dir, '%d_samples.npz' % (samples_len))
                print('saving {} samples to {} ...'.format(samples_len, intermediate_samples_path))
                np.savez(intermediate_samples_path, samples_np=samples_np)
                save_interval_counter += 1

        print('saving {} samples to {} ...'.format(samples_len, latest_samples_path))
        np.savez(latest_samples_path, samples_np=samples_np)
