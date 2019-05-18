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

def maybe_download_and_extract(data_dir, url='http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'):
    if not os.path.exists(os.path.join(data_dir, 'cifar-10-batches-py')):
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        filename = url.split('/')[-1]
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                    float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()
            filepath, _ = urllib.request.urlretrieve(url, filepath, _progress)
            print()
            statinfo = os.stat(filepath)
            print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
            tarfile.open(filepath, 'r:gz').extractall(data_dir)

def unpickle(file):
    fo = open(file, 'rb')
    if (sys.version_info >= (3, 0)):
        import pickle
        d = pickle.load(fo, encoding='latin1')
    else:
        import cPickle
        d = cPickle.load(fo)
    fo.close()
    return {'x': d['data'].reshape((10000,3,32,32)), 'y': np.array(d['labels']).astype(np.uint8)}

def load(data_dir, subset='train'):
    maybe_download_and_extract(data_dir)
    if subset=='train':
        train_data = [unpickle(os.path.join(data_dir,'cifar-10-batches-py','data_batch_' + str(i))) for i in range(1,6)]
        trainx = np.concatenate([d['x'] for d in train_data],axis=0)
        trainy = np.concatenate([d['y'] for d in train_data],axis=0)
        return trainx, trainy
    elif subset=='test':
        test_data = unpickle(os.path.join(data_dir,'cifar-10-batches-py','test_batch'))
        testx = test_data['x']
        testy = test_data['y']
        return testx, testy
    else:
        raise NotImplementedError('subset should be either train or test')

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
        self.data, self.labels = load(os.path.join(data_dir,'cifar-10-python'), subset=subset)
        print(self.data.shape)
        print(self.labels.shape)
        self.num_classes = 10
        self.class2labels = {} # n,
        self.class2data = {} # n, 3, h, w
        for i in range(self.num_classes):
            indices = np.where(self.labels==i)
            self.class2labels[i] = self.labels[indices]
            self.class2data[i] = np.transpose(self.data[indices], (0,2,3,1)) # (N,3,32,32) -> (N,32,32,3)

        self.batches_labels = []
        self.batches_data = []

        self.p = 0 # pointer to where we are in iteration
        self.rng = np.random.RandomState(1) if rng is None else rng

    def get_observation_size(self):
        return self.class2data[0].shape[1:]

    def get_num_labels(self):
        return sum([len(arrs) for arrs in self.class2labels.values()]) + 1

    def reset(self):
        self.p = 0

    def __iter__(self):
        return self

    def __next__(self, n=None):
        """ n is the number of examples to fetch """
        if n is not None:
            raise('n should be none according to ria.')
        if n is None: n = self.batch_size

        # on first iteration lazily permute all data
        if self.p == 0 and self.shuffle:
            for i in range(self.num_classes):
                n = len(self.class2labels[i])
                inds = self.rng.permutation(n)
                i_labels_shuffled = self.class2labels[i][inds]
                i_data_shuffled = self.class2data[i][inds]
                for batch_start in list(range(0,n,batch_size))[:-1]:
                    self.batches_labels.append(i_labels_shuffled[batch_start:batch_start+batch_size])        
                    self.batches_data.append(i_data_shuffled[batch_start:batch_start+batch_size])
        # inds = list(range(len(self.batches_labels)))
        # random.shuffle(inds)
        combined = list(zip(self.batches_labels, self.batches_data))
        random.shuffle(combined)
        self.batches_labels[:], self.batches_data[:] = zip(*combined)

        # on last iteration reset the counter and raise StopIteration
        if self.p >= len(self.batches_labels):
            self.reset() # reset for next time we get called
            raise StopIteration

        # on intermediate iterations fetch the next batch
        x = self.batches_data[self.p]
        y = self.batches_labels[self.p]
        self.p += 1

        if self.return_labels:
            return x,y
        else:
            return x

    next = __next__  # Python 2 compatibility (https://stackoverflow.com/questions/29578469/how-to-make-an-object-both-a-python2-and-python3-iterator)

# Testing sorted data loader
if __name__ == "__main__":
    batch_size = 16
    nr_gpu = 8
    train_data = DataLoader('/tmp/pcnn-pp_data', 'train', batch_size * nr_gpu, rng=None, shuffle=True, return_labels=True)