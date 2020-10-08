"""
Utilities for downloading and unpacking the CIFAR-10 dataset, originally published
by Krizhevsky et al. and hosted here: https://www.cs.toronto.edu/~kriz/cifar.html
"""

import os
import sys
import tarfile
from six.moves import urllib
import numpy as np
import random
from PIL import Image
# from data.data_scripts.data_utils import *

def show_batch(imgs):
    num, h, w, _ = imgs.shape
    Image.fromarray(imgs.reshape(num*h, w, 3)).show()

class DataLoader(object):
    """ an object that generates batches of CIFAR-10 data for training """

    def __init__(self, data_dir, data_set, subset, batch_size, rng=None, shuffle=False, return_labels=False, custom_load_str=None, **kwargs):
        """
        - data_dir is location where to store files
        - subset is train|test
        - batch_size is int, of #examples to load at once
        - rng is np.random.RandomState object for reproducibility
        """

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.return_labels = return_labels

        loaded = np.load(os.path.join(data_dir, data_set+'.npz'))
        data = loaded['trainx'] if subset == 'train' else loaded['testx']
        labels = loaded['trainy'] if subset == 'train' else loaded['testy']
        self.num_classes = np.max(labels) + 1

        print('sorting...')
        self.class2labels = {} # n,
        self.class2data = {}
        for i in range(self.num_classes):
            indices = np.where(labels==i)
            self.class2labels[i] = labels[indices]
            self.class2data[i] = data[indices]
            # np.savez('cifar-class{}'.format(i), trainx=self.class2data[i], trainy = self.class2labels[i])

        self.p = 0 # pointer to where we are in iteration
        self.rng = np.random.RandomState(1) if rng is None else rng

        self.batches_labels = []
        self.batches_data = []
        self.reset(force_shuffle=True)
        # print('batch shape', self.batches_data[0].shape, self.batches_labels[0].shape)

    def get_observation_size(self):
        return self.class2data[0].shape[1:]

    def get_num_labels(self):
        return self.num_classes

    def reset(self, force_shuffle=False):
        self.p = 0
        # lazily permute all data
        if self.shuffle or force_shuffle:
            # create new batches
            # print('fresh batches...')
            self.batches_labels = []
            self.batches_data = []
            for i in range(self.num_classes):
                n = len(self.class2labels[i])
                inds = self.rng.permutation(n)
                i_labels_shuffled = self.class2labels[i][inds]
                i_data_shuffled = self.class2data[i][inds]
                for batch_start in list(range(0,n,self.batch_size))[:-1]:
                    self.batches_labels.append(i_labels_shuffled[batch_start:batch_start+self.batch_size])
                    self.batches_data.append(i_data_shuffled[batch_start:batch_start+self.batch_size])

            # shuffle batches
            # print('shuffling...')
            combined = list(zip(self.batches_labels, self.batches_data))
            random.shuffle(combined)
            self.batches_labels[:], self.batches_data[:] = zip(*combined)

    def __iter__(self):
        return self

    def __next__(self, n=None):
        """ n is the number of examples to fetch """
        assert(n is None or n % self.batch_size==0)
        if n is None:
            n = self.batch_size

        num_batches = n // self.batch_size
        # print(n, num_batches)


        # # on last iteration reset the counter and raise StopIteration
        # if self.p + n > self.data.shape[0]:
        #     raise StopIteration
        #
        # # on intermediate iterations fetch the next batch
        # x = self.data[self.p: self.p + n]
        # y = self.labels[self.p: self.p + n]
        # self.p += self.batch_size

        # on last iteration reset the counter and raise StopIteration
        if self.p + num_batches - 1 >= len(self.batches_labels):
            raise StopIteration

        # on intermediate iterations fetch the next batch
        x = self.batches_data[self.p:self.p + num_batches]
        x = np.concatenate(x)
        y = self.batches_labels[self.p:self.p + num_batches]
        y = np.concatenate(y)
        self.p += num_batches

        if self.return_labels:
            return x,y
        else:
            return x

    next = __next__  # Python 2 compatibility (https://stackoverflow.com/questions/29578469/how-to-make-an-object-both-a-python2-and-python3-iterator)

# Testing sorted data loader
if __name__ == "__main__":
    batch_size = 16
    nr_gpu = 1
    train_data = DataLoader('../../data/', 'fashion_3', 'train', batch_size * nr_gpu, rng=None, shuffle=True, return_labels=True)
    print(train_data.next()[0].shape)
    show_batch(train_data.next()[0])