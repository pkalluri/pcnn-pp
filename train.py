"""
Trains a Pixel-CNN++ generative model with specified loss function.
The model only trains (evaluates+backprops) on the specified loss, indicated by the --accumulator flag
but the model evaluates several additional loss functions for the sole purpose of printing those losses as additional information.
Uses multiple GPUs, indicated by the flag --nr_gpu.

Example usage:
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_double_cnn.py --nr_gpu 4

TODO: This code is broken for the conditional case.
"""

import os
import sys
import json
import argparse
import time
import pickle

import numpy as np
import tensorflow as tf

from pixel_cnn_pp import nn
from pixel_cnn_pp.model import model_spec
from utils import images as images_util
from utils import plotting_losses as losses_util
from utils import io as io_util
import data_loaders as data_loaders

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
# data I/O
parser.add_argument('-i', '--data_dir', type=str, default='data', help='Location for the dataset')
parser.add_argument('-d', '--data_set', type=str, default='cifar', help='eg cifar, imagenet, etc')
parser.add_argument('-p', '--params_dir', type=str, default=None, help='Optional: pretrained parameters to load.')
# parser.add_argument('-r', '--params_dir', dest='params_dir', action='store_true', help='Restore training from previous model checkpoint?')
parser.add_argument('-o', '--save_dir', type=str, default=None, help='Location for parameter checkpoints and samples')
parser.add_argument('-t', '--sample_frequency', type=int, default=20, help='Every how many epochs to write checkpoint/samples?')
parser.add_argument('-tag', '--tag', type=str, default='', help='Tag to add to end of save directory (e.g. DEBUG to mark a debugging run')
# model
parser.add_argument('-q', '--nr_resnet', type=int, default=5, help='Number of residual blocks per stage of the model')
parser.add_argument('-n', '--nr_filters', type=int, default=160, help='Number of filters to use across the model. Higher = larger model.')
parser.add_argument('-m', '--nr_logistic_mix', type=int, default=10, help='Number of logistic components in the mixture. Higher = more flexible model')
parser.add_argument('-z', '--resnet_nonlinearity', type=str, default='concat_elu', help='Which nonlinearity to use in the ResNet layers. One of "concat_elu", "elu", "relu" ')
parser.add_argument('-c', '--class_conditional', dest='class_conditional', action='store_true', help='Condition generative model on labels?')
# optimization
parser.add_argument('-l', '--learning_rate', type=float, default=0.001, help='Base learning rate')
parser.add_argument('-e', '--lr_decay', type=float, default=0.999995, help='Learning rate decay, applied every step of the optimization')
parser.add_argument('-b', '--batch_size', type=int, default=16, help='Batch size during training per GPU')
parser.add_argument('-u', '--init_batch_size', type=int, default=None, help='How much data to use for data-dependent initialization.')
parser.add_argument('-dp', '--dropout_p', type=float, default=0.5, help='Dropout strength (i.e. 1 - keep_prob). 0 = No dropout, higher = more dropout.')
parser.add_argument('-x', '--max_epochs', type=int, default=5000, help='How many epochs to run in total?')
parser.add_argument('-g', '--nr_gpu', type=int, default=1, help='How many GPUs to distribute the training across?') # Note: original default was 8
parser.add_argument('-ed', '--energy_distance', dest='energy_distance', action='store_true', help='use energy distance in place of likelihood')
parser.add_argument('-en', '--entropy', dest='entropy', action='store_true', help='Include -entropy term in loss, encouraging diversity?')
parser.add_argument('-a', '--accumulator', type=str, default='standard', help='How to accumulate many samples losses into one batch loss')
# evaluation
parser.add_argument('--polyak_decay', type=float, default=0.9995, help='Exponential decay rate of the sum of previous model iterates during Polyak averaging')
parser.add_argument('-ns', '--n_batches_sampled', type=int, default=2, help='How many batches of samples to output.')
# reproducibility
parser.add_argument('-r', '--seed', type=int, default=1, help='Random seed to use')
args = parser.parse_args()
if not args.init_batch_size: # unspecified
    args.init_batch_size = args.batch_size * args.nr_gpu # TODO this line hasn't been tested for nr_gpu > 1
print('input args:\n', json.dumps(vars(args), indent=4, separators=(',',':'))) # pretty print args

helpful_description = f'{args.data_set}_{args.nr_filters}L_{args.accumulator}_B{args.batch_size}'
args.save_dir += f'_{helpful_description}'
if args.tag: 
    args.save_dir+=f'_{args.tag}'

if args.entropy:
    raise('entropy is not implemented.')
# energy distance or maximum likelihood?
if args.energy_distance:
    raise("This is not compatible with the log-sum loss.")
    loss_fun = nn.energy_distance # todo: this is currently broken, because it does not take the same args as the following loss
else:
    loss_fun = nn.discretized_mix_logistic_loss


# fix random seed for reproducibility
rng = np.random.RandomState(args.seed)
tf.set_random_seed(args.seed)

# --------------------------------LOADING DATA---------------------------------------------

# TODO data loading can be way cleaner.

# initialize data loaders for train/test splits
# datasets we know how to download
if args.data_set in ['cifar', 'cifar_sorted', 'svhn', 'svhn_sorted', 'shapes', 'shapes_sorted', 'dbg', 'imagenet']:# datasets we know how to download
    if args.data_set == 'cifar':
        import data_loaders.cifar10_data as cifar10_data
        DataLoader = cifar10_data.DataLoader
    elif args.data_set == 'cifar_sorted':
        import data_loaders.cifar10_sorted_data as cifar10_sorted_data
        DataLoader = cifar10_sorted_data.DataLoader
    elif args.data_set == 'svhn':
        import data_loaders.svhn as svhn
        DataLoader = svhn.DataLoader
    elif args.data_set == 'svhn_sorted':
        import data_loaders.svhn as svhn_sorted
        DataLoader = svhn_sorted.DataLoader
    elif args.data_set == 'shapes':
        import data_loaders.shapes as shapes
        DataLoader = shapes.DataLoader
    elif args.data_set == 'shapes_sorted':
        import data_loaders.shapes_sorted as shapes_sorted
        DataLoader = shapes_sorted.DataLoader
    elif args.data_set == 'dbg':
        import data_loaders.dbg as dbg
        DataLoader = dbg.DataLoader
    elif args.data_set == 'imagenet':
        if args.class_conditional:
            raise("We currently don't have labels for the small imagenet data set")
        import data_loaders.imagenet_data as imagenet_data
        DataLoader = imagenet_data.DataLoader
    else:
        raise("We currently do not know how to download the specified dataset. You can convert it to an npz yourself and give the name of it.")
    train_data = DataLoader(args.data_dir, 'train', args.batch_size * args.nr_gpu, rng=rng, shuffle=True, return_labels=args.class_conditional)
    test_data = DataLoader(args.data_dir, 'test', args.batch_size * args.nr_gpu, shuffle=False, return_labels=args.class_conditional)
    image_dims = train_data.get_observation_size() # e.g. a tuple (32,32,3)
# elif args.data_set.startswith('cifar'):
#     # one class of cifar
#     import data_loaders.cifar10_class_data as cifar10_class_data
#     DataLoader = cifar10_class_data.DataLoader
#     which_class = int(args.data_set.split('cifar')[1])
#     train_data = DataLoader(args.data_dir, which_class, 'train', args.batch_size * args.nr_gpu, rng=rng, shuffle=True, return_labels=args.class_conditional)
#     test_data = DataLoader(args.data_dir, which_class, 'test', args.batch_size * args.nr_gpu, shuffle=False, return_labels=args.class_conditional)
#     image_dims = train_data.get_observation_size() # e.g. a tuple (32,32,3)
else: # from a particular file
    if args.class_conditional:
        raise("We currently only know how to load UNCONDITIONAL custom datasets.")
    import data_loaders.npz_data as npz_data
    DataLoader = npz_data.DataLoader
    train_data = DataLoader(args.data_dir, args.data_set, 'train', args.batch_size * args.nr_gpu, rng=rng, shuffle=True, return_labels=args.class_conditional)
    test_data = DataLoader(args.data_dir, args.data_set, 'test', args.batch_size * args.nr_gpu, shuffle=False, return_labels=args.class_conditional)
    image_dims = train_data.get_observation_size() # e.g. a tuple (32,32,3)
# assert len(image_dims) == 3, 'We currently assume multi-channel images'


# ------------------------------------DEFINING DATA PREPROCESSING-----------------------------------------
#

# define converting numpy inputs into feed_dict for use with tensorflow
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

#
# ------------------------------------DEFINING MODEL-----------------------------------------

# data place holders
x_init = tf.placeholder(tf.float32, shape=(args.init_batch_size,) + image_dims)
xs = [tf.placeholder(tf.float32, shape=(args.batch_size, ) + image_dims) for i in range(args.nr_gpu)]

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
model_opt = { 'nr_resnet': args.nr_resnet, 'nr_filters': args.nr_filters, 'nr_logistic_mix': args.nr_logistic_mix, 'resnet_nonlinearity': args.resnet_nonlinearity, 'energy_distance': args.energy_distance }
model = tf.make_template('model', model_spec)

# run once for data dependent initialization of parameters
init_pass = model(x_init, h_init, init=True, dropout_p=args.dropout_p, **model_opt)

# keep track of moving average
all_params = tf.trainable_variables()
ema = tf.train.ExponentialMovingAverage(decay=args.polyak_decay)
maintain_averages_op = tf.group(ema.apply(all_params))
ema_params = [ema.average(p) for p in all_params]

# get loss gradients over multiple GPUs + sampling
grads = []
loss_gen = []
loss_gen_test = []
loss_gen_standard = []
loss_gen_sum = []
loss_gen_max = []
loss_gen_standard_test = []
loss_gen_sum_test = []
loss_gen_max_test = []
new_x_gen = []
for i in range(args.nr_gpu):
    with tf.device('/gpu:%d' % i):
        # train
        out = model(xs[i], hs[i], ema=None, dropout_p=args.dropout_p, **model_opt)
        loss_gen.append(loss_fun(tf.stop_gradient(xs[i]), out, args.accumulator, args.entropy))
        loss_gen_standard.append(loss_fun(tf.stop_gradient(xs[i]), out, "log_prod", args.entropy))
        loss_gen_sum.append(loss_fun(tf.stop_gradient(xs[i]), out, "log_sum", args.entropy))
        loss_gen_max.append(loss_fun(tf.stop_gradient(xs[i]), out, "log_max", args.entropy))

        # get gradients
        grads.append(tf.gradients(loss_gen[i], all_params, colocate_gradients_with_ops=True))

        # test
        out = model(xs[i], hs[i], ema=ema, dropout_p=0., **model_opt)
        loss_gen_test.append(loss_fun(xs[i], out, args.accumulator, args.entropy))
        loss_gen_standard_test.append(loss_fun(xs[i], out, "log_prod", args.entropy))
        loss_gen_sum_test.append(loss_fun(xs[i], out, "log_sum", args.entropy))
        loss_gen_max_test.append(loss_fun(xs[i], out, "log_max", args.entropy))

        # sample
        out = model(xs[i], h_sample[i], ema=ema, dropout_p=0, **model_opt)
        if args.energy_distance:
            new_x_gen.append(out[0])
        else:
            new_x_gen.append(nn.sample_from_discretized_mix_logistic(out, args.nr_logistic_mix))

# add losses and gradients together and get model updates
tf_lr = tf.placeholder(tf.float32, shape=[])
with tf.device('/gpu:0'):
    for i in range(1,args.nr_gpu):
        loss_gen[0] += loss_gen[i]
        loss_gen_test[0] += loss_gen_test[i]

        loss_gen_standard[0] += loss_gen_standard[i]
        loss_gen_sum[0] += loss_gen_sum[i]
        loss_gen_max[0] += loss_gen_max[i]

        loss_gen_standard_test[0] += loss_gen_standard_test[i]
        loss_gen_sum_test[0] += loss_gen_sum_test[i]
        loss_gen_max_test[0] += loss_gen_max_test[i]

        for j in range(len(grads[0])):
            grads[0][j] += grads[i][j]
    # training op
    optimizer = tf.group(nn.adam_updates(all_params, grads[0], lr=tf_lr, mom1=0.95, mom2=0.9995), maintain_averages_op)

# convert loss to bits/dim
bits_per_dim = loss_gen[0]/(args.nr_gpu*np.log(2.)*np.prod(image_dims)*args.batch_size)
bits_per_dim_test = loss_gen_test[0]/(args.nr_gpu*np.log(2.)*np.prod(image_dims)*args.batch_size)
bits_per_dim_standard = loss_gen_standard[0]/(args.nr_gpu*np.log(2.)*np.prod(image_dims)*args.batch_size)
bits_per_dim_standard_test = loss_gen_standard_test[0]/(args.nr_gpu*np.log(2.)*np.prod(image_dims)*args.batch_size)
bits_per_dim_sum = loss_gen_sum[0]/(args.nr_gpu*np.log(2.)*np.prod(image_dims)*args.batch_size)
bits_per_dim_sum_test = loss_gen_sum_test[0]/(args.nr_gpu*np.log(2.)*np.prod(image_dims)*args.batch_size)
bits_per_dim_max = loss_gen_max[0]/(args.nr_gpu*np.log(2.)*np.prod(image_dims)*args.batch_size)
bits_per_dim_max_test = loss_gen_max_test[0]/(args.nr_gpu*np.log(2.)*np.prod(image_dims)*args.batch_size)
loss_names = ['standard', 'sum', 'max']

# define sampling from the model
def sample_from_model(sess):
    x_gen = [np.zeros((args.batch_size,) + image_dims, dtype=np.float32) for i in range(args.nr_gpu)]
    for yi in range(image_dims[0]):
        for xi in range(image_dims[1]):
            new_x_gen_np = sess.run(new_x_gen, {xs[i]: x_gen[i] for i in range(args.nr_gpu)})
            for i in range(args.nr_gpu):
                x_gen[i][:,yi,xi,:] = new_x_gen_np[i][:,yi,xi,:]
    return np.concatenate(x_gen, axis=0)

# init & save
initializer = tf.global_variables_initializer()
saver = tf.train.Saver()

# ----------------------------------TRAINING-------------------------------------------

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
lr = args.learning_rate
with tf.Session() as sess:

    train_data.reset()  # set the iterator to 0
    if args.params_dir:
        ckpt_file = args.params_dir + '/params_' + args.data_set + '.ckpt'
        print('restoring parameters from', ckpt_file)
        saver.restore(sess, ckpt_file)
    else:
        print('initializing the model...')
        sess.run(initializer)
        feed_dict = make_feed_dict(train_data.next(args.init_batch_size), init=True)  # manually retrieve exactly init_batch_size examples
        sess.run(init_pass, feed_dict)
    
    train_losses_log = {loss_name:[] for loss_name in loss_names}
    test_losses_log = {loss_name:[] for loss_name in loss_names}
    all_epochs_samples = [] 
    last_samples_path = None

    # save initial train and test losses
    # TODO move to separate function (so we can call once here and once in for loop)
    # train losses
    train_losses = []
    train_standard_losses = []
    train_sum_losses = []
    train_max_losses = []
    for d in train_data:
        feed_dict = make_feed_dict(d)
        # forward pass, backward pass, and update model on each gpu
        lr *= args.lr_decay
        feed_dict.update({ tf_lr: lr })
        train_loss, standard_loss, sum_loss, max_loss = sess.run([bits_per_dim, bits_per_dim_standard, bits_per_dim_sum, bits_per_dim_max], feed_dict)
        # TODO below could be rewritten using a dictionary, which would be more extensible to new loss functions
        train_losses.append(train_loss)
        train_standard_losses.append(standard_loss)
        train_sum_losses.append(sum_loss)
        train_max_losses.append(max_loss)
    total_train_loss = np.mean(train_losses)
    total_train_standard_loss = np.mean(train_standard_losses)
    total_train_sum_loss = np.mean(train_sum_losses)
    total_train_max_loss = np.mean(train_max_losses)

    # test losses
    test_losses = []
    test_standard_losses = []
    test_sum_losses = []
    test_max_losses = []
    for d in test_data:
        feed_dict = make_feed_dict(d)
        test_loss, standard_loss, sum_loss, max_loss = sess.run([bits_per_dim_test, bits_per_dim_standard_test, bits_per_dim_sum_test, bits_per_dim_max_test], feed_dict)
        test_losses.append(test_loss)
        test_standard_losses.append(standard_loss)
        test_sum_losses.append(sum_loss)
        test_max_losses.append(max_loss)
    total_test_loss = np.mean(test_losses)
    total_test_standard_loss = np.mean(test_standard_losses)
    total_test_sum_loss = np.mean(test_sum_losses)
    total_test_max_loss = np.mean(test_max_losses)

    # log progress to console
    curr_train_losses = {'training_loss':total_train_loss, 'standard':total_train_standard_loss, 'sum':total_train_sum_loss, 'max':total_train_max_loss}
    curr_test_losses =  {'training_loss':total_test_loss, 'standard':total_test_standard_loss, 'sum':total_test_sum_loss, 'max':total_test_max_loss}
    # TODO: currently this nice dictionary is not being used immediately below.
    print("Iteration %s, time = %ds, \
            train bits_per_dim = %.8f, \
            test bits_per_dim = %.8f, \
            standard_train bits_per_dim = %.8f, \
            sum_train bits_per_dim = %.8f, \
            max_train bits_per_dim = %.8f, \
            standard_test bits_per_dim = %.8f, \
            sum_test bits_per_dim = %.8f, \
            max_test bits_per_dim = %.8f, \
            " % ('init', 0, total_train_loss, total_test_loss,
            total_train_standard_loss, total_train_sum_loss, total_train_max_loss,
            total_test_standard_loss, total_test_sum_loss, total_test_max_loss,))
    sys.stdout.flush()
    train_losses_log = {name:log+[curr_train_losses[name]] for name,log in train_losses_log.items()}
    test_losses_log = {name:log+[curr_test_losses[name]] for name,log in test_losses_log.items()}        
   
    print('starting 0th epoch')
    for epoch in range(args.max_epochs):
        if epoch % args.sample_frequency == 0:
            print('saving')
            saver.save(sess, os.path.join(args.save_dir, 'params.ckpt')) # save params
            all_losses_log = {'train':train_losses_log, 'test': test_losses_log}

            io_util.save_pickle(all_losses_log, os.path.join(args.save_dir, 'training_losses.pickle'))
            losses_util.save_plot(
                all_losses_log, 
                os.path.join(args.save_dir, 'training_losses_separate.png'),
                title=helpful_description,
                separate=True
                )
            losses_util.save_plot(
                all_losses_log, 
                os.path.join(args.save_dir, 'training_losses.png'),
                title=helpful_description,
                separate=False
                )

            # generate samples from the model
            n_samples = args.n_batches_sampled * args.batch_size
            sampled_batches = []
            for i in range(args.n_batches_sampled):
                sampled_batches.append(sample_from_model(sess)) # generate a batch, shape of batch: batch_size x H x W x C
            samples = np.concatenate(sampled_batches,axis=0) # shape: n_batches*batch_size x H x W x C
            all_epochs_samples.append(samples)

            # save latest samples as small png
            tiled_samples = images_util.tile(samples) # shape: M*H x MxW
            samples_path = os.path.join(args.save_dir,f'latest_samples-epoch{epoch}.png')
            images_util.save_with_title(tiled_samples, title=f'{helpful_description}\nepoch {epoch}', path=samples_path)
            if last_samples_path is not None: os.remove(last_samples_path)
            last_samples_path = samples_path

            # save all training samples as summary npz and png
            np.savez(os.path.join(args.save_dir, 'training_samples.npz'), samples=np.stack(all_epochs_samples)) # shape: N_(SAVED)EPOCHS x N_GENERATED_SAMPLES x H x W x C
            # savez so that this is easily extensible to save classes array, losses array, etc, too
            all_samples_tiled = images_util.tile(np.concatenate(all_epochs_samples), grid_shape=(int((epoch / args.sample_frequency)+1), n_samples))
            all_samples_path = os.path.join(args.save_dir,'training_samples.png')
            images_util.save_with_title(all_samples_tiled, title=f'{helpful_description}\nepochs 0-{epoch}', path=all_samples_path)

        print('training')
        begin = time.time()

        if epoch != 0:
            train_data.reset()  # rewind the iterator back to 0

        # train losses
        train_losses = []
        train_standard_losses = []
        train_sum_losses = []
        train_max_losses = []
        for d in train_data:
            feed_dict = make_feed_dict(d)
            # forward pass, backward pass, and update model on each gpu
            lr *= args.lr_decay
            feed_dict.update({ tf_lr: lr })
            train_loss, standard_loss, sum_loss, max_loss, _ = sess.run([bits_per_dim, bits_per_dim_standard, bits_per_dim_sum, bits_per_dim_max, optimizer], feed_dict)
            # TODO below could be rewritten using a dictionary, which would be more extensible to new loss functions
            train_losses.append(train_loss)
            train_standard_losses.append(standard_loss)
            train_sum_losses.append(sum_loss)
            train_max_losses.append(max_loss)
        total_train_loss = np.mean(train_losses)
        total_train_standard_loss = np.mean(train_standard_losses)
        total_train_sum_loss = np.mean(train_sum_losses)
        total_train_max_loss = np.mean(train_max_losses)

        # test losses
        test_losses = []
        test_standard_losses = []
        test_sum_losses = []
        test_max_losses = []
        for d in test_data:
            feed_dict = make_feed_dict(d)
            test_loss, standard_loss, sum_loss, max_loss = sess.run([bits_per_dim_test, bits_per_dim_standard_test, bits_per_dim_sum_test, bits_per_dim_max_test], feed_dict)
            test_losses.append(test_loss)
            test_standard_losses.append(standard_loss)
            test_sum_losses.append(sum_loss)
            test_max_losses.append(max_loss)
        total_test_loss = np.mean(test_losses)
        total_test_standard_loss = np.mean(test_standard_losses)
        total_test_sum_loss = np.mean(test_sum_losses)
        total_test_max_loss = np.mean(test_max_losses)

        # log progress to console
        curr_train_losses = {'training_loss':total_train_loss, 'standard':total_train_standard_loss, 'sum':total_train_sum_loss, 'max':total_train_max_loss}
        curr_test_losses =  {'training_loss':total_test_loss, 'standard':total_test_standard_loss, 'sum':total_test_sum_loss, 'max':total_test_max_loss}
        # TODO: currently this nice dictionary is not being used immediately below.
        print("Iteration %d, time = %ds, \
                train bits_per_dim = %.8f, \
                test bits_per_dim = %.8f, \
                standard_train bits_per_dim = %.8f, \
                sum_train bits_per_dim = %.8f, \
                max_train bits_per_dim = %.8f, \
                standard_test bits_per_dim = %.8f, \
                sum_test bits_per_dim = %.8f, \
                max_test bits_per_dim = %.8f, \
                " % (epoch, time.time()-begin, total_train_loss, total_test_loss,
                total_train_standard_loss, total_train_sum_loss, total_train_max_loss,
                total_test_standard_loss, total_test_sum_loss, total_test_max_loss,))
        sys.stdout.flush()
        train_losses_log = {name:log+[curr_train_losses[name]] for name,log in train_losses_log.items()}
        test_losses_log = {name:log+[curr_test_losses[name]] for name,log in test_losses_log.items()}        

