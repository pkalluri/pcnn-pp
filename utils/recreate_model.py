"""
Given a Tensorflow generative model, outputs its inception score
"""

import os
import sys
import json
import argparse
import time

import numpy as np
import tensorflow as tf

from pixel_cnn_pp import nn
from pixel_cnn_pp.model import model_spec
from utils import plotting, args_reading
from scoring import inception

def recreate_model(args, batch_size_generator):
    # fix random seed for reproducibility
    rng = np.random.RandomState(args.seed)
    tf.set_random_seed(args.seed)

    # energy distance or maximum likelihood?
    if args.energy_distance:
        loss_fun = nn.energy_distance # todo: this is currently broken, because it does not take the same args as the following loss
    else:
        loss_fun = nn.discretized_mix_logistic_loss

    # initialize data loaders for train/test splits
    if args.data_set in ['imagenet', 'cifar','cifar_sorted']:
        if args.data_set == 'imagenet' and args.class_conditional:
            raise("We currently don't have labels for the small imagenet data set")
        if args.data_set == 'cifar':
            import data.cifar10_data as cifar10_data
            DataLoader = cifar10_data.DataLoader
        elif args.data_set == 'cifar_sorted':
            import data.cifar10_sorted_data as cifar10_sorted_data
            DataLoader = cifar10_sorted_data.DataLoader
        elif args.data_set == 'imagenet':
            import data.imagenet_data as imagenet_data
            DataLoader = imagenet_data.DataLoader
        train_data = DataLoader(args.data_dir, 'train', args.batch_size * args.nr_gpu, rng=rng, shuffle=True, return_labels=args.class_conditional)
        # test_data = DataLoader(args.data_dir, 'test', args.batch_size * args.nr_gpu, shuffle=False, return_labels=args.class_conditional)
        obs_shape = train_data.get_observation_size() # e.g. a tuple (32,32,3)
    elif args.data_set.startswith('cifar'):
        # one class of cifar
        import data.cifar10_class_data as cifar10_class_data
        DataLoader = cifar10_class_data.DataLoader
        which_class = int(args.data_set.split('cifar')[1])
        train_data = DataLoader(args.data_dir, which_class, 'train', args.batch_size * args.nr_gpu, rng=rng, shuffle=True, return_labels=args.class_conditional)
        test_data = DataLoader(args.data_dir, which_class, 'test', args.batch_size * args.nr_gpu, shuffle=False, return_labels=args.class_conditional)
        obs_shape = train_data.get_observation_size() # e.g. a tuple (32,32,3)
    else:
        # if args.class_conditional:
        #     raise("This is an unconditional dataset.")
        import data.npz_data as from_file_data
        DataLoader = from_file_data.DataLoader
        train_data = DataLoader(args.data_dir, args.data_set, 'train', args.batch_size * args.nr_gpu, rng=rng, shuffle=True, return_labels=args.class_conditional)
        # test_data = DataLoader(args.data_dir, args.data_set, 'test', args.batch_size * args.nr_gpu, shuffle=False, return_labels=args.class_conditional)
        obs_shape = train_data.get_observation_size() # e.g. a tuple (32,32,3)
    # assert len(obs_shape) == 3, 'assumed right now'

    # data place holders
    print("creating data place holders...")
    x_init = tf.placeholder(tf.float32, shape=(batch_size_generator,) + obs_shape)
    xs = [tf.placeholder(tf.float32, shape=(batch_size_generator, ) + obs_shape) for i in range(args.nr_gpu)]

    # if the model is class-conditional we'll set up label placeholders + one-hot encodings 'h' to condition on
    if args.class_conditional:
        print("creating label placeholders...")
        num_labels = train_data.get_num_labels()
        y_init = tf.placeholder(tf.int32, shape=(batch_size_generator,))
        h_init = tf.one_hot(y_init, num_labels)
        y_sample = np.split(np.mod(np.arange(batch_size_generator*args.nr_gpu), num_labels), args.nr_gpu)
        h_sample = [tf.one_hot(tf.Variable(y_sample[i], trainable=False), num_labels) for i in range(args.nr_gpu)]
        ys = [tf.placeholder(tf.int32, shape=(batch_size_generator,)) for i in range(args.nr_gpu)]
        hs = [tf.one_hot(ys[i], num_labels) for i in range(args.nr_gpu)]
    else:
        h_init = None
        h_sample = [None] * args.nr_gpu
        hs = h_sample

    # create the model
    print("creating model...")
    model_opt = { 'nr_resnet': args.nr_resnet, 'nr_filters': args.nr_filters, 'nr_logistic_mix': args.nr_logistic_mix, 'resnet_nonlinearity': args.resnet_nonlinearity, 'energy_distance': args.energy_distance }
    model = tf.make_template('model', model_spec)

    # run once for data dependent initialization of parameters
    print("running init_pass...")
    init_pass = model(x_init, h_init, init=True, dropout_p=args.dropout_p, **model_opt)

    # keep track of moving average
    all_params = tf.trainable_variables()
    ema = tf.train.ExponentialMovingAverage(decay=args.polyak_decay)
    maintain_averages_op = tf.group(ema.apply(all_params))
    # ema_params = [ema.average(p) for p in all_params]

    # # get loss gradients over multiple GPUs + sampling
    grads = []
    loss_gen = []
    loss_gen_test = []
    print("getting sample generation functions on gpu...")
    new_x_gen = []
    for i in range(args.nr_gpu):
        with tf.device('/gpu:%d' % i):
            # train
            out = model(xs[i], hs[i], ema=None, dropout_p=args.dropout_p, **model_opt)
            loss_gen.append(loss_fun(tf.stop_gradient(xs[i]), out, args.accumulator, args.entropy))

            # gradients
            grads.append(tf.gradients(loss_gen[i], all_params, colocate_gradients_with_ops=True))

            # test
            out = model(xs[i], hs[i], ema=ema, dropout_p=0., **model_opt)
            loss_gen_test.append(loss_fun(xs[i], out, args.accumulator, args.entropy))

            # sample
            out = model(xs[i], h_sample[i], ema=ema, dropout_p=0, **model_opt)
            if args.energy_distance:
                new_x_gen.append(out[0])
            else:
                if args.sampling_variance is not None:
                    new_x_gen.append(nn.sample_from_narrow_discretized_mix_logistic(out, args.nr_logistic_mix, var=args.sampling_variance))
                else:
                    new_x_gen.append(nn.sample_from_discretized_mix_logistic(out, args.nr_logistic_mix))

    # add losses and gradients together and get training updates
    tf_lr = tf.placeholder(tf.float32, shape=[])
    with tf.device('/gpu:0'):
        for i in range(1,args.nr_gpu):
            loss_gen[0] += loss_gen[i]
            loss_gen_test[0] += loss_gen_test[i]
            for j in range(len(grads[0])):
                grads[0][j] += grads[i][j]
        # training op
        # optimizer = tf.group(nn.adam_updates(all_params, grads[0], lr=tf_lr, mom1=0.95, mom2=0.9995), maintain_averages_op)

    # convert loss to bits/dim
    #bits_per_dim = loss_gen[0]/(args.nr_gpu*np.log(2.)*np.prod(obs_shape)*args.batch_size_generator)
    #bits_per_dim_test = loss_gen_test[0]/(args.nr_gpu*np.log(2.)*np.prod(obs_shape)*args.batch_size_generator)

    # init & save
    print("generating initializer and saver...")
    initializer = tf.global_variables_initializer()
    saver = tf.train.Saver()

    return saver, obs_shape, new_x_gen, xs
