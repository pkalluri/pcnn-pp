"""
Utilities for loading a dataset from an npz file.
(Can use script png_to_npz.py to create the npz files)
"""

import os
import sys
import tarfile
from six.moves import urllib

import numpy as np
from imageio import imread

class DataLoader(object):
    """ an object that generates batches of given dataset """

    def __init__(self, data_dir, data_set, subset, batch_size, rng=None, shuffle=False, return_labels=False, custom_load_str=None, **kwargs):
        """ 
        - data_dir is location of the dir where the data is stored
        - subset is train|test 
        - batch_size is int, of #examples to load at once
        - rng is np.random.RandomState object for reproducibility
        """

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.return_labels = return_labels

        loaded = np.load(os.path.join(data_dir, data_set+'.npz'))

        print("custom_load_str:", custom_load_str)
        if custom_load_str:
            self.data = loaded[custom_load_str]
        else:
            self.data = loaded['trainx'] if subset == 'train' else loaded['testx']
        
        if self.return_labels:
            self.labels = loaded['trainy'] if subset == 'train' else loaded['testy']
        
        self.p = 0 # pointer to where we are in iteration
        self.rng = np.random.RandomState(1) if rng is None else rng

    def get_observation_size(self):
        return self.data.shape[1:]

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
            if self.return_labels:
                self.labels = self.labels[inds]

        # on last iteration reset the counter and raise StopIteration
        if self.p + n > self.data.shape[0]:
            self.reset() # reset for next time we get called
            raise StopIteration

        # on intermediate iterations fetch the next batch
        x = self.data[self.p : self.p + n]
        if self.return_labels:
            y = self.labels[self.p : self.p + n]
        self.p += self.batch_size

        if self.return_labels:
            return x,y
        else:
            return x

    next = __next__  # Python 2 compatibility (https://stackoverflow.com/questions/29578469/how-to-make-an-object-both-a-python2-and-python3-iterator)