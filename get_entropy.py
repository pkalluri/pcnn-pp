"""
Trains a Pixel-CNN++ generative model on CIFAR-10 or Tiny ImageNet data.
Uses multiple GPUs, indicated by the flag --nr_gpu

Example usage:
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_double_cnn.py --nr_gpu 4
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

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-args', '--args_file', type=str, default='', help='.out file to parse arguments from and overwrite any of the below')
#parser.add_argument('-args', '--args_file', type=str, default='save/713925.out', help='.out file to parse arguments from and overwrite any of the below')
#parser.add_argument('-args', '--args_file', type=str, default='save/714767.out', help='out file to parse arguments from and overwrite any of the below')
#parser.add_argument('-args', '--args_file', type=str, default='save_neurips/716587.out', help='out file to parse arguments from and overwrite any of the below')
#parser.add_argument('-args', '--args_file', type=str, default='', help='out file to parse arguments from and overwrite any of the below')
#parser.add_argument('-o', '--checkpoint_dir', type=str, default='save/_cifar_st_samples', help='Directory where the checkpoint files (and possibly samples) live')
#parser.add_argument('-o', '--checkpoint_dir', type=str, default='save/_713925_samples', help='Directory where the checkpoint files (and possibly samples) live')
#parser.add_argument('-o', '--checkpoint_dir', type=str, default='save/_714767_samples', help='Directory where the checkpoint files (and possibly samples) live')
#parser.add_argument('-o', '--checkpoint_dir', type=str, default='save_neurips/_716587_samples', help='Directory where the checkpoint files (and possibly samples) live')
parser.add_argument('-o', '--checkpoint_dir', type=str, default='save/_697740', help='Directory where the checkpoint files (and possibly samples) live')
parser.add_argument('-cp', '--checkpoint_prefix', type=str, default='params_cifar.ckpt', help='Checkpoint files prefix')
# data I/O
parser.add_argument('-i', '--data_dir', type=str, default='save/_697740', help='Location for the dataset')
# parser.add_argument('-o', '--save_dir', type=str, default='save_neurips', help='Location for parameter checkpoints and samples')
#parser.add_argument('-d', '--data_set', type=str, default='all_samples_from_params_cifar.ckpt', help='npz file to read from')
parser.add_argument('-d', '--data_set', type=str, default='samples_from_params_cifar.ckpt', help='npz file to read from')
#parser.add_argument('-t', '--save_interval', type=int, default=5, help='Every how many epochs to write checkpoint/samples?')
#parser.add_argument('-r', '--load_params', dest='load_params', action='store_true', help='Restore training from previous model checkpoint?')
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
parser.add_argument('-l', '--learning_rate', type=float, default=0.001, help='Base learning rate')
parser.add_argument('-e', '--lr_decay', type=float, default=0.999995, help='Learning rate decay, applied every step of the optimization')
parser.add_argument('-b', '--batch_size', type=int, default=16, help='Batch size during training per GPU')
parser.add_argument('-u', '--init_batch_size', type=int, default=16, help='How much data to use for data-dependent initialization.')
parser.add_argument('-p', '--dropout_p', type=float, default=0.5, help='Dropout strength (i.e. 1 - keep_prob). 0 = No dropout, higher = more dropout.')
parser.add_argument('-x', '--max_epochs', type=int, default=5000, help='How many epochs to run in total?')
parser.add_argument('-g', '--nr_gpu', type=int, default=8, help='How many GPUs to distribute the training across?')
# evaluation
parser.add_argument('--polyak_decay', type=float, default=0.9995, help='Exponential decay rate of the sum of previous model iterates during Polyak averaging')
parser.add_argument('-ns', '--num_samples', type=int, default=1, help='How many batches of samples to output.')
# reproducibility
parser.add_argument('-s', '--seed', type=int, default=1, help='Random seed to use')
# new
parser.add_argument('-cs', '--custom_load_string', type=str, default='samples_np', help='Load string for custom dataset')
args = parser.parse_args()

if args.entropy:
    raise('entropy is not implemented.')

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

args = parser.parse_args()
if args.args_file:
    overwrite_args = read_args_from_out_file(args.args_file)
    d = vars(args)
    d.update(overwrite_args)
print('input args:\n', json.dumps(vars(args), indent=4, separators=(',',':'))) # pretty print args
# -----------------------------------------------------------------------------

# fix random seed for reproducibility
rng = np.random.RandomState(args.seed)
tf.set_random_seed(args.seed)

# energy distance or maximum likelihood?
if args.energy_distance:
    loss_fun = nn.energy_distance # todo: this is currently broken, because it does not take the same args as the following loss
else:
    loss_fun = nn.discretized_mix_logistic_loss

# initialize data loaders for train/test splits
print('intializing data loaders...')
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
    # test_data = DataLoader(args.data_dir, which_class, 'test', args.batch_size * args.nr_gpu, shuffle=False, return_labels=args.class_conditional)
    obs_shape = train_data.get_observation_size() # e.g. a tuple (32,32,3)
else:
    # if args.class_conditional:
    #     raise("This is an unconditional dataset.")
    import data.npz_data as from_file_data
    print('from_file_data.DataLoader')
    DataLoader = from_file_data.DataLoader
    train_data = DataLoader(args.data_dir, args.data_set, 'train', args.batch_size * args.nr_gpu, rng=rng, shuffle=True, return_labels=args.class_conditional, custom_load_str=args.custom_load_string)
    # test_data = DataLoader(args.data_dir, args.data_set, 'test', args.batch_size * args.nr_gpu, shuffle=False, return_labels=args.class_conditional)
    obs_shape = train_data.get_observation_size() # e.g. a tuple (32,32,3)
# assert len(obs_shape) == 3, 'assumed right now'

# data place holders
print('data place holders...')
x_init = tf.placeholder(tf.float32, shape=(args.init_batch_size,) + obs_shape)
xs = [tf.placeholder(tf.float32, shape=(args.batch_size, ) + obs_shape) for i in range(args.nr_gpu)]

# if the model is class-conditional we'll set up label placeholders + one-hot encodings 'h' to condition on
if args.class_conditional:
    num_labels = train_data.get_num_labels()
    y_init = tf.placeholder(tf.int32, shape=(args.init_batch_size,))
    h_init = tf.one_hot(y_init, num_labels)
    y_sample = np.split(np.mod(np.arange(args.batch_size*args.nr_gpu), num_labels), args.nr_gpu)
    h_sample = [tf.one_hot(tf.Variable(y_sample[i], trainable=False), num_labels) for i in range(args.nr_gpu)]
    ys = [tf.placeholder(tf.int32, shape=(args.batch_size,)) for i in range(args.nr_gpu)]
    hs = [tf.one_hot(ys[i], num_labels) for i in range(args.nr_gpu)]
else:
    h_init = None
    h_sample = [None] * args.nr_gpu
    hs = h_sample

# create the model
print('create model...')
model_opt = { 'nr_resnet': args.nr_resnet, 'nr_filters': args.nr_filters, 'nr_logistic_mix': args.nr_logistic_mix, 'resnet_nonlinearity': args.resnet_nonlinearity, 'energy_distance': args.energy_distance }
model = tf.make_template('model', model_spec)

# run once for data dependent initialization of parameters
print('init pass...')
init_pass = model(x_init, h_init, init=True, dropout_p=args.dropout_p, **model_opt)

# loss gen
print('create loss gen...')
loss_gen = []
for i in range(args.nr_gpu):
    with tf.device('/gpu:%d' % i):
        out = model(xs[i], hs[i], ema=None, dropout_p=args.dropout_p, **model_opt)
        loss_gen.append(loss_fun(tf.stop_gradient(xs[i]), out, "log_prod", args.entropy))

# add losses
print('add losses...')
tf_lr = tf.placeholder(tf.float32, shape=[])
with tf.device('/gpu:0'):
    for i in range(1,args.nr_gpu):
        loss_gen[0] += loss_gen[i]

# convert loss to bits/dim
bits_per_dim = loss_gen[0]/(args.nr_gpu*np.log(2.)*np.prod(obs_shape)*args.batch_size)

# init & save
print('init and save...')
initializer = tf.global_variables_initializer()
saver = tf.train.Saver()

# turn numpy inputs into feed_dict for use with tensorflow
def make_feed_dict(data, init=False):
    if type(data) is tuple:
        x,y = data
    else:
        x = data
        y = None
    x = np.cast[np.float32]((x - 127.5) / 127.5) # input to pixelCNN is scaled from uint8 [0,255] to float in range [-1,1]
    if init:
        feed_dict = {x_init: x}
        if y is not None:
            feed_dict.update({y_init: y})
    else:
        x = np.split(x, args.nr_gpu)
        feed_dict = {xs[i]: x[i] for i in range(args.nr_gpu)}
        if y is not None:
            y = np.split(y, args.nr_gpu)
            feed_dict.update({ys[i]: y[i] for i in range(args.nr_gpu)})
    return feed_dict

# get entropy
print('getting entropy...')
with tf.Session() as sess:
    ckpt_file = os.path.join(args.checkpoint_dir, args.checkpoint_prefix)
    # ckpt_file = (os.path.join(args.checkpoint_dir, 'params_' + args.data_set + '.ckpt')
    print('restoring parameters from', ckpt_file)
    saver.restore(sess, ckpt_file)

    print('getting entropy from {} generated samples...'.format(len(train_data.data)))
    train_losses = []
    for d in train_data:
        feed_dict = make_feed_dict(d)
        train_loss = sess.run([bits_per_dim], feed_dict)
        train_losses.append(train_loss)
    train_loss_gen = np.mean(train_losses)
    print("Entropy: {}".format(train_loss_gen))
