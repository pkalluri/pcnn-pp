"""
Utilities for downloading and unpacking the SVHN dataset
"""

import os
import sys
import tarfile
from six.moves import urllib
import numpy as np
import random
import urllib.request
from scipy.io import loadmat

def download_if_not_downloaded(svhn_dir, filename):
    if not os.path.exists(svhn_dir):
        os.makedirs(svhn_dir)
        print(f'created {svhn_dir}.')
    url = os.path.join('http://ufldl.stanford.edu/housenumbers', filename) 
    filepath = os.path.join(svhn_dir, filename)
    if not os.path.exists(filepath):
        urllib.request.urlretrieve(url, filepath)
        print(f'created {filepath}')

def load_downloaded_svhn(svhn_dir, filename):
    "loads as numpy arrays: xs as shape (N, 32, 32, 3) and ys as shape (N,)"
    filepath = os.path.join(svhn_dir, filename)
    data = loadmat(filepath)
    x, y = data['X'], data['y']
    x = np.transpose(x, (3, 0, 1, 2)) # from (d, d, c, n) to (n, d, d, c)
    y = np.squeeze(y)
    print(type(x), type(y))
    print(x.shape, y.shape)
    return x, y

def load_svhn(data_dir, subset='train'):
    assert(subset in ('train', 'test'))
    svhn_dir = os.path.join(data_dir, 'SVHN')
    filename = f'{subset}_32x32.mat'
    download_if_not_downloaded(svhn_dir, filename)
    x, y = load_downloaded_svhn(svhn_dir, filename)
    return x, y # trainx should be (N,32,32,3), trainy should be (N,)

class DataLoader(object):
    """ an object that generates batches of SVHN data for training """

    def __init__(self, data_dir, subset, batch_size, rng=None, shuffle=False, return_labels=False):
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

        # create temporary storage for data, if not yet created
        if not os.path.exists(data_dir):
            print('creating folder', data_dir)
            os.makedirs(data_dir)
        print('loading...')
        # load SVHN training data to RAM. data should be (N,32,32,3), labels should be (N,)
        data, labels = load_svhn(data_dir, subset=subset)

        print('sorting...')
        self.num_classes = 10
        self.class2labels = {} # n,
        self.class2data = {} # n, 3, h, w
        for i in range(self.num_classes):
            indices = np.where(labels==i)
            self.class2labels[i] = labels[indices]
            self.class2data[i] = data[indices] # (N,32,32,3)
            # np.savez('cifar-class{}'.format(i), trainx=self.class2data[i], trainy = self.class2labels[i])

        self.p = 0 # pointer to where we are in iteration
        self.rng = np.random.RandomState(1) if rng is None else rng

        self.batches_labels = []
        self.batches_data = []
        self.reset(force_shuffle=True)
        # print('batch shape', self.batches_data[0].shape, self.batches_labels[0].shape)

    def get_observation_size(self):
        # return self.data.shape[1:]
        # print(self.class2data[0].shape[1:])
        return self.class2data[0].shape[1:]

    def get_num_labels(self):
        # return np.amax(self.labels) + 1
        return 10

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
    nr_gpu = 2
    train_data = DataLoader('../../data', 'train', batch_size * nr_gpu, rng=None, shuffle=True, return_labels=True)
    imgs, labels = train_data.next()
    print(labels)

    # show img
    from PIL import Image
    img = Image.fromarray(imgs[0]) # from numpy array to image
    img.show()
