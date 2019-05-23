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
from utils import plotting
from scoring import inception

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ov', '--overwrite_samples', dest='overwrite_samples', action='store_true', help='Overwrite generated files?')
    #parser.add_argument('-args', '--args_file', type=str, default='', help='.out file to parse arguments from and overwrite any of the below')
    #parser.add_argument('-args', '--args_file', type=str, default='save/713925.out', help='.out file to parse arguments from and overwrite any of the below')
    #parser.add_argument('-args', '--args_file', type=str, default='save/714767.out', help='out file to parse arguments from and overwrite any of the below')
    #parser.add_argument('-args', '--args_file', type=str, default='save_neurips/716587.out', help='out file to parse arguments from and overwrite any of the below')
    parser.add_argument('-args', '--args_file', type=str, default='', help='out file to parse arguments from and overwrite any of the below')
    #parser.add_argument('-args', '--args_file', type=str, default='', help='out file to parse arguments from and overwrite any of the below')
    #parser.add_argument('-o', '--checkpoint_dir', type=str, default='save/_cifar_st_samples', help='Directory where the checkpoint files (and possibly samples) live')
    #parser.add_argument('-o', '--checkpoint_dir', type=str, default='save/_713925_samples', help='Directory where the checkpoint files (and possibly samples) live')
    #parser.add_argument('-o', '--checkpoint_dir', type=str, default='save/_714767_samples', help='Directory where the checkpoint files (and possibly samples) live')
    #parser.add_argument('-o', '--checkpoint_dir', type=str, default='save_neurips/_716587_samples', help='Directory where the checkpoint files (and possibly samples) live')
    parser.add_argument('-o', '--checkpoint_dir', type=str, default='../data/', help='Directory where the checkpoint files (and possibly samples) live')
    #parser.add_argument('-o', '--checkpoint_dir', type=str, default='save/_697740', help='Directory where the checkpoint files (and possibly samples) live')
    parser.add_argument('-cp', '--checkpoint_prefix', type=str, default=None, help='Checkpoint files prefix')
    # parser.add_argument('-ss', '--save_samples', type=bool, default=True, help='Whether or not to save generated samples')
    parser.add_argument('-nbg', '--num_batches_generated', type=int, default=100, help='How many batches of samples to generate')
    parser.add_argument('-b', '--batch_size_generator', type=int, default=10, help='Batch size for generation')
    parser.add_argument('-u', '--init_batch_size', type=int, default=16, help='How much data to use for data-dependent initialization.')
    parser.add_argument('-nsp', '--num_splits', type=int, default=1, help='How many splits to use for inception score')
    parser.add_argument('-i', '--data_dir', type=str, default='/tmp/pcnn-pp_data', help='Location for the dataset')
    # Below only used for graph definition
    # --------------------------------
    #parser.add_argument('-i', '--data_dir', type=str, default='../data', help='Location for the dataset')
    #parser.add_argument('-o', '--save_dir', type=str, default='save', help='Location for parameter checkpoints and samples')
    parser.add_argument('-d', '--data_set', type=str, default='cifar', help='Can be either cifar|imagenet')
    #parser.add_argument('-t', '--save_interval', type=int, default=20, help='Every how many epochs to write checkpoint/samples?')
    parser.add_argument('-r', '--load_params', dest='load_params', action='store_true', help='Restore training from previous model checkpoint?')
    # model
    parser.add_argument('-q', '--nr_resnet', type=int, default=5, help='Number of residual blocks per stage of the model')
    parser.add_argument('-n', '--nr_filters', type=int, default=160, help='Number of filters to use across the model. Higher = larger model.')
    parser.add_argument('-m', '--nr_logistic_mix', type=int, default=10, help='Number of logistic components in the mixture. Higher = more flexible model')
    parser.add_argument('-z', '--resnet_nonlinearity', type=str, default='concat_elu', help='Which nonlinearity to use in the ResNet layers. One of "concat_elu", "elu", "relu" ')
    parser.add_argument('-c', '--class_conditional', dest='class_conditional', action='store_true', help='Condition generative model on labels?')
    parser.add_argument('-ed', '--energy_distance', dest='energy_distance', action='store_true', help='use energy distance in place of likelihood')
    parser.add_argument('-en', '--entropy', dest='entropy', action='store_true', help='Include -entropy term in loss, encouraging diversity?')
    parser.add_argument('-a', '--accumulator', type=str, default='standard', help='How to accumulate many samples losses into one batch loss')
    # optimization
    #parser.add_argument('-l', '--learning_rate', type=float, default=0.001, help='Base learning rate')
    #parser.add_argument('-e', '--lr_decay', type=float, default=0.999995, help='Learning rate decay, applied every step of the optimization')
    parser.add_argument('-p', '--dropout_p', type=float, default=0.5, help='Dropout strength (i.e. 1 - keep_prob). 0 = No dropout, higher = more dropout.')
    #parser.add_argument('-x', '--max_epochs', type=int, default=5000, help='How many epochs to run in total?')
    parser.add_argument('-g', '--nr_gpu', type=int, default=8, help='How many GPUs to distribute the training across?')
    # evaluation
    parser.add_argument('--polyak_decay', type=float, default=0.9995, help='Exponential decay rate of the sum of previous model iterates during Polyak averaging')
    # parser.add_argument('-ns', '--num_samples', type=int, default=1, help='How many batches of samples to output.')
    # reproducibility
    parser.add_argument('-s', '--seed', type=int, default=1, help='Random seed to use')
    # --------------------------------
    args = parser.parse_args()
    if args.args_file:
        overwrite_args = read_args_from_out_file(args.args_file)
        d = vars(args)
        d.update(overwrite_args)
    print('input args:\n', json.dumps(vars(args), indent=4, separators=(',',':'))) # pretty print args
    return args

def read_args_from_out_file(filename):
    with open(filename, 'r') as f:
        json_string = ''
        reading_json = False
        for line in f:
            if reading_json:
                json_string += line.strip()
                if line.startswith('}'):
                    reading_json = False

            if line.startswith('input args:'):
                reading_json = True
    
    data = json.loads(json_string)
    return data

def recreate_model(args):
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
        test_data = DataLoader(args.data_dir, 'test', args.batch_size * args.nr_gpu, shuffle=False, return_labels=args.class_conditional)
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
        test_data = DataLoader(args.data_dir, args.data_set, 'test', args.batch_size * args.nr_gpu, shuffle=False, return_labels=args.class_conditional)
        obs_shape = train_data.get_observation_size() # e.g. a tuple (32,32,3)
    # assert len(obs_shape) == 3, 'assumed right now'

    # data place holders
    print("creating data place holders...")
    x_init = tf.placeholder(tf.float32, shape=(args.batch_size_generator,) + obs_shape)
    xs = [tf.placeholder(tf.float32, shape=(args.batch_size_generator, ) + obs_shape) for i in range(args.nr_gpu)]

    # if the model is class-conditional we'll set up label placeholders + one-hot encodings 'h' to condition on
    if args.class_conditional:
        print("creating label placeholders...")
        num_labels = train_data.get_num_labels()
        y_init = tf.placeholder(tf.int32, shape=(args.batch_size_generator,))
        h_init = tf.one_hot(y_init, num_labels)
        y_sample = np.split(np.mod(np.arange(args.batch_size_generator*args.nr_gpu), num_labels), args.nr_gpu)
        h_sample = [tf.one_hot(tf.Variable(y_sample[i], trainable=False), num_labels) for i in range(args.nr_gpu)]
        ys = [tf.placeholder(tf.int32, shape=(args.batch_size_generator,)) for i in range(args.nr_gpu)]
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
    ema_params = [ema.average(p) for p in all_params]

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
        optimizer = tf.group(nn.adam_updates(all_params, grads[0], lr=tf_lr, mom1=0.95, mom2=0.9995), maintain_averages_op)

    # convert loss to bits/dim
    #bits_per_dim = loss_gen[0]/(args.nr_gpu*np.log(2.)*np.prod(obs_shape)*args.batch_size_generator)
    #bits_per_dim_test = loss_gen_test[0]/(args.nr_gpu*np.log(2.)*np.prod(obs_shape)*args.batch_size_generator)

    # init & save
    print("generating initializer and saver...")
    initializer = tf.global_variables_initializer()
    saver = tf.train.Saver()

    return saver, obs_shape, new_x_gen, xs

def sample_from_model(sess, obs_shape, new_x_gen, xs):
    x_gen = [np.zeros((args.batch_size_generator,) + obs_shape, dtype=np.float32) for i in range(args.nr_gpu)]
    for yi in range(obs_shape[0]):
        for xi in range(obs_shape[1]):
            new_x_gen_np = sess.run(new_x_gen, {xs[i]: x_gen[i] for i in range(args.nr_gpu)})
            sys.stdout.write(".")
            sys.stdout.flush()
            for i in range(args.nr_gpu):
                x_gen[i][:,yi,xi,:] = new_x_gen_np[i][:,yi,xi,:]
    print()
    return np.concatenate(x_gen, axis=0)

def get_inception_scores_and_write_predictions(samples, num_splits, pred_path):
    print('getting inception score on {} samples with {} splits...'.format(len(samples), num_splits))
    process = lambda img: ((img+1)*255/2).astype('uint8')
    samples = [process(s) for s in samples]
    mean, var, preds = inception.get_inception_score(samples, splits=args.num_splits)
    print('inception score: mean={}, variance={}'.format(mean, var))

    print('saving predictions to {} ...'.format(pred_path))
    np.savez(pred_path, preds=preds)

if __name__ == "__main__":
    args = get_args()

    if args.checkpoint_prefix:
        ckpt_file = args.checkpoint_prefix
    else:
        ckpt_file = 'params_' + args.data_set + '.ckpt'

    # ckpt_path = os.path.join(args.checkpoint_dir, ckpt_file)
    # all_samples_path = os.path.join(args.checkpoint_dir,'all_samples_from_%s.npz' % (ckpt_file))
    # all_preds_path = os.path.join(args.checkpoint_dir,'all_preds_on_samples_from_%s.npz' % (ckpt_file))
    all_samples_path = os.path.join(args.checkpoint_dir,'cifar.npz')
    all_preds_path = os.path.join(args.checkpoint_dir,'preds_cifar.npz')

    if not args.overwrite_samples and os.path.exists(all_samples_path):
        print('loading samples from {}'.format(all_samples_path))
        samples_np = np.load(all_samples_path)['trainx']
        #samples_np = np.load(all_samples_path)['samples_np']
        samples_list = list(samples_np)
        print('loaded {} samples'.format(len(samples_list)))
    else:
        saver, obs_shape, new_x_gen, xs = recreate_model(args)
        with tf.Session() as sess:
            print('restoring parameters from {} ...'.format(ckpt_path))
            saver.restore(sess, ckpt_path)

            # TODO split up saved batches into multiple files to avoid OOM, maybe
            print('generating {} batches of {} samples...'.format(args.num_batches_generated, args.batch_size_generator))
            samples = []
            save_interval = 10
            save_interval_counter = 1
            for i in range(args.num_batches_generated):
                print('generating batch {}...'.format(i))
                samples.append(sample_from_model(sess, obs_shape, new_x_gen, xs))

                samples_np = np.concatenate(samples,axis=0)
                samples_list = list(samples_np)
                print(len(samples_list), save_interval_counter * save_interval)

                if len(samples_list) >= (save_interval_counter * save_interval):
                    intermediate_samples_path = os.path.join(args.checkpoint_dir,'%d_samples_from_%s.npz' % (len(samples_list), ckpt_file))
                    print('saving samples to {} ...'.format(intermediate_samples_path))
                    np.savez(intermediate_samples_path, samples_np=samples_np)
                    samples_for_pred = list(samples_np)

                    intermediate_pred_path = os.path.join(args.checkpoint_dir,'%d_preds_on_samples_from_%s.npz' % (len(samples_list), ckpt_file))
                    get_inception_scores_and_write_predictions(samples_list, args.num_splits, intermediate_pred_path)

                    save_interval_counter += 1

        print('saving samples to {} ...'.format(all_samples_path))
        np.savez(all_samples_path, samples_np=samples_np)

    get_inception_scores_and_write_predictions(samples_list, args.num_splits, all_preds_path)
