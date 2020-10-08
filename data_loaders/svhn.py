"""
Utilities for downloading and unpacking the CIFAR-10 dataset, originally published
by Krizhevsky et al. and hosted here: https://www.cs.toronto.edu/~kriz/cifar.html
"""

import os
import sys
import tarfile
from six.moves import urllib
import numpy as np
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
    return x, y

def load_svhn(data_dir, subset='train'):
    assert(subset in ('train', 'test'))
    svhn_dir = os.path.join(data_dir, 'SVHN')
    filename = f'{subset}_32x32.mat'
    download_if_not_downloaded(svhn_dir, filename)
    x, y = load_downloaded_svhn(svhn_dir, filename)
    return x, y # trainx should be (N,32,32,3), trainy should be (N,)

class DataLoader(object):
    """ an object that generates batches of CIFAR-10 data for training """

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

        # create temporary storage for the data, if not yet created
        if not os.path.exists(data_dir):
            print('creating folder', data_dir)
            os.makedirs(data_dir)

        # load CIFAR-10 training data to RAM
        self.data, self.labels = load_svhn(data_dir, subset=subset)
        #np.savez('../../data/cifar.npz', trainx=self.data)
        
        self.p = 0 # pointer to where we are in iteration
        self.rng = np.random.RandomState(1) if rng is None else rng

    def get_observation_size(self):
        return self.data.shape[1:]

    def get_num_labels(self):
        return np.amax(self.labels) + 1

    def reset(self):
        self.p = 0

    def __iter__(self):
        return self

    def __next__(self, n=None):
        """ n is the number of examples to fetch """
        if n is None: n = self.batch_size

        # on first iteration lazily permute all data
        if self.p == 0 and self.shuffle:
            inds = self.rng.permutation(self.data.shape[0])
            self.data = self.data[inds]
            self.labels = self.labels[inds]

        # on last iteration reset the counter and raise StopIteration
        if self.p + n > self.data.shape[0]:
            self.reset() # reset for next time we get called
            raise StopIteration

        # on intermediate iterations fetch the next batch
        x = self.data[self.p : self.p + n]
        y = self.labels[self.p : self.p + n]
        self.p += self.batch_size

        if self.return_labels:
            return x,y
        else:
            return x

    next = __next__  # Python 2 compatibility (https://stackoverflow.com/questions/29578469/how-to-make-an-object-both-a-python2-and-python3-iterator)

# Testing data loader
if __name__ == "__main__":
    batch_size = 16
    nr_gpu = 8
    train_data = DataLoader('../../data', 'train', batch_size * nr_gpu, rng=None, shuffle=True, return_labels=True)
    imgs,labels = train_data.next()
    print(labels)

    # show img
    from PIL import Image
    # img = Image.fromarray(imgs[0]) # from numpy array to image
    # img.show()
    for i in range(10):
        Image.fromarray(imgs[i]).show()

