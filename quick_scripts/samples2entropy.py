"""
Given a Tensorflow generative model and the samples that it generates, get its entropy
"""

import os
import json
import argparse

import numpy as np
import tensorflow as tf

from utils.training_args import out_to_updated_args
from pixel_cnn_pp import nn
from pixel_cnn_pp.model import model_spec

def get_args():
    parser = argparse.ArgumentParser()
    # The underscores are added so that none of the args will be overwritten by the args from the .out file

    # Oft-changed parameters
    parser.add_argument('-pd', '--params_dir_', required=True, type=str, help='The directory where the checkpoint parameters live. Example: save/pretrained')
    parser.add_argument('-p', '--samples_path_', required=True, type=str, default=None, help='The filepath to our samples')

    # Defaulted parameters
    parser.add_argument('-a', '--args_file_', type=str, default=None, help='The .out file to parse arguments from for graph generation')
    parser.add_argument('-cp', '--checkpoint_prefix_', type=str, default=None, help='Checkpoint files prefix - ex. cifar_params.cpkt')
    parser.add_argument('-cs', '--custom_load_string_', type=str, default='samples_np', help='Load string for dataset')

    args = parser.parse_args()

    if args.args_file_:
        args_file = args.args_file_
    else:
        args_file = args.params_dir_ + '.out'

    args = out_to_updated_args(args_file, args)

    print('input args:\n', json.dumps(vars(args), indent=4, separators=(',',':'))) # pretty print args
    return args


if __name__ == "__main__":
    args = get_args()

    entropy = False
    nr_gpu = 1

    samples_dir, samples_dataset = os.path.split(args.samples_path_)
    samples_dataset = samples_dataset.rstrip('.npz')
    print('samples dir: {}, samples dataset: {}'.format(samples_dir, samples_dataset))

    # fix random seed for reproducibility
    rng = np.random.RandomState(args.seed)
    tf.set_random_seed(args.seed)

    loss_fun = nn.discretized_mix_logistic_loss

    # initialize data loaders for train/test splits
    print('initializing data loader...')
    import data.npz_data as from_file_data
    DataLoader = from_file_data.DataLoader
    train_data = DataLoader(samples_dir, samples_dataset, 'train', args.batch_size * nr_gpu, rng=rng,
                            shuffle=True, return_labels=args.class_conditional,
                            custom_load_str=args.custom_load_string_)
    obs_shape = train_data.get_observation_size()  # e.g. a tuple (32,32,3)

    # data place holders
    print('data place holders...')
    x_init = tf.placeholder(tf.float32, shape=(args.init_batch_size,) + obs_shape)
    xs = [tf.placeholder(tf.float32, shape=(args.batch_size,) + obs_shape) for i in range(nr_gpu)]

    # if the model is class-conditional we'll set up label placeholders + one-hot encodings 'h' to condition on
    if args.class_conditional:
        num_labels = train_data.get_num_labels()
        y_init = tf.placeholder(tf.int32, shape=(args.init_batch_size,))
        h_init = tf.one_hot(y_init, num_labels)
        y_sample = np.split(np.mod(np.arange(args.batch_size * nr_gpu), num_labels), nr_gpu)
        h_sample = [tf.one_hot(tf.Variable(y_sample[i], trainable=False), num_labels) for i in range(nr_gpu)]
        ys = [tf.placeholder(tf.int32, shape=(args.batch_size,)) for i in range(nr_gpu)]
        hs = [tf.one_hot(ys[i], num_labels) for i in range(nr_gpu)]
    else:
        h_init = None
        h_sample = [None] * nr_gpu
        hs = h_sample

    # create the model
    print('create model...')
    model_opt = {'nr_resnet': args.nr_resnet, 'nr_filters': args.nr_filters, 'nr_logistic_mix': args.nr_logistic_mix,
                 'resnet_nonlinearity': args.resnet_nonlinearity, 'energy_distance': args.energy_distance}
    model = tf.make_template('model', model_spec)

    # run once for data dependent initialization of parameters
    print('init pass...')
    init_pass = model(x_init, h_init, init=True, dropout_p=args.dropout_p, **model_opt)

    # loss gen
    print('create loss gen...')
    loss_gen = []
    for i in range(nr_gpu):
        with tf.device('/gpu:%d' % i):
            out = model(xs[i], hs[i], ema=None, dropout_p=args.dropout_p, **model_opt)
            loss_gen.append(loss_fun(tf.stop_gradient(xs[i]), out, "log_prod", entropy))

    # add losses
    print('add losses...')
    tf_lr = tf.placeholder(tf.float32, shape=[])
    with tf.device('/gpu:0'):
        for i in range(1, nr_gpu):
            loss_gen[0] += loss_gen[i]

    # convert loss to bits/dim
    bits_per_dim = loss_gen[0] / (nr_gpu * np.log(2.) * np.prod(obs_shape) * args.batch_size)

    # init & save
    print('init and save...')
    initializer = tf.global_variables_initializer()
    saver = tf.train.Saver()


    # turn numpy inputs into feed_dict for use with tensorflow
    def make_feed_dict(data, init=False):
        if type(data) is tuple:
            x, y = data
        else:
            x = data
            y = None
        x = np.cast[np.float32](
            (x - 127.5) / 127.5)  # input to pixelCNN is scaled from uint8 [0,255] to float in range [-1,1]
        if init:
            feed_dict = {x_init: x}
            if y is not None:
                feed_dict.update({y_init: y})
        else:
            x = np.split(x, nr_gpu)
            feed_dict = {xs[i]: x[i] for i in range(nr_gpu)}
            if y is not None:
                y = np.split(y, nr_gpu)
                feed_dict.update({ys[i]: y[i] for i in range(nr_gpu)})
        return feed_dict

    if args.checkpoint_prefix_:
        ckpt_prefix = args.checkpoint_prefix_
    else:
        ckpt_prefix = 'params_' + args.data_set + '.ckpt'

    ckpt_path = os.path.join(args.params_dir_, ckpt_prefix)

    with tf.Session() as sess:
        print('restoring parameters from', ckpt_path)
        saver.restore(sess, ckpt_path)

        print('getting loss from {} generated samples...'.format(len(train_data.data)))
        train_losses = []
        for d in train_data:
            feed_dict = make_feed_dict(d)
            train_loss = sess.run([bits_per_dim], feed_dict)
            train_losses.append(train_loss)

        print('getting entropy from 1000 losses...'.format(len(train_data.data)))
        train_losses = train_losses[:1000]
        train_loss_gen = np.mean(train_losses)
        print("Entropy: {}".format(train_loss_gen))
