"Official code to calculate Inception Score"
# Adapted from https://github.com/openai/improved-gan/blob/master/inception_score/model.py
# They adapted from tensorflow/tensorflow/models/image/imagenet/classify_image.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

# Vanilla
FastGFile = tf.gfile.FastGFile
GraphDef = tf.GraphDef

# other attempts to get the write import:

# FastGFile = tf.compat.v1.gfile.FastGFile
# GraphDef = tf.compat.v1.GraphDef

# import tensorboard as tb
# tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

# import tensorflow.compat.v1 as tf 
# tf.disable_v2_behavior()

import glob
import scipy.misc
import math
import sys

MODEL_DIR = '../imagenet_model'
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
softmax = None # If None, updated (see end of file)

def predictions_to_scores(predictions: np.ndarray, splits: int=10):
  scores = []
  for i in range(splits):
    part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
    kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
    kl = np.mean(np.sum(kl, 1))
    scores.append(np.exp(kl))
  return np.mean(scores), np.std(scores)


def samples_to_predictions(samples: np.ndarray, batchsize: int=1):
  assert(type(samples) == np.ndarray)
  assert(len(samples.shape) == 4) # shape of samples: N_SAMPLES X W x H X C
  # assert(np.max(images[0]) > 10)
  # assert(np.min(images[0]) >= 0.0)
  # for img in images:
  #   img = img.astype(np.float32)
  #   samples.append(np.expand_dims(img, 0))
  preds = []
  with tf.Session() as sess:
    n_batches = int(math.ceil(float(len(samples)) / float(batchsize)))
    for i in range(n_batches):
        sys.stdout.write(".")
        batch = samples[(i * batchsize):((i+1) * batchsize)] # shape: B x W x H x C
        pred = sess.run(softmax, {'ExpandDims:0': batch}) # shape: B x N_CLASSES
        preds.append(pred)
  preds = np.concatenate(preds, 0) # shape: N_SAMPLES x N_CLASSES
  print(preds.shape)
  return preds


# Call this function with list of images. Each of elements should be a 
# numpy array with values ranging from 0 to 255.
def get_inception_score(images, splits=10):
    preds = get_inception_preds(images)
    mean, var = get_inception_score_from_preds(preds, splits)
    return mean, var, preds


def get_inception_preds(images):
  assert(type(images) == list)
  assert(type(images[0]) == np.ndarray)
  assert(len(images[0].shape) == 3)
  assert(np.max(images[0]) > 10)
  assert(np.min(images[0]) >= 0.0)
  samples = []
  for img in images:
    img = img.astype(np.float32)
    samples.append(np.expand_dims(img, 0))
  batchsize = 1
  with tf.Session() as sess:
    preds = []
    n_batches = int(math.ceil(float(len(samples)) / float(batchsize)))
    for i in range(n_batches):
        sys.stdout.write(".")
        sys.stdout.flush()
        batch = samples[(i * batchsize):min((i + 1) * batchsize, len(samples))]
        batch = np.concatenate(batch, 0)
        pred = sess.run(softmax, {'ExpandDims:0': batch})
        preds.append(pred)
    preds = np.concatenate(preds, 0)
    return preds

def get_inception_score_from_preds(preds, splits=10):
    scores = []
    for i in range(splits):
      part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
      kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
      kl = np.mean(np.sum(kl, 1))
      scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)

# This function is called automatically.
def _init_inception():
  global softmax
  if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(MODEL_DIR, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(MODEL_DIR)
  with FastGFile(os.path.join(
      MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')
  # Works with an arbitrary minibatch size.
  with tf.Session() as sess:
    pool3 = sess.graph.get_tensor_by_name('pool_3:0')
    ops = pool3.graph.get_operations()
    for op_idx, op in enumerate(ops):
        for o in op.outputs:
            shape = o.get_shape()
            shape = [s.value for s in shape]
            new_shape = []
            for j, s in enumerate(shape):
                if s == 1 and j == 0:
                    new_shape.append(None)
                else:
                    new_shape.append(s)
            o.set_shape(tf.TensorShape(new_shape))
    w = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
    logits = tf.matmul(tf.squeeze(pool3, [1, 2]), w)
    softmax = tf.nn.softmax(logits)

if softmax is None:
  _init_inception()