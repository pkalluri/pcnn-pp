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
        print('loading...')
        # load CIFAR-10 training data to RAM
        data, labels = load(os.path.join(data_dir,'cifar-10-python'), subset=subset)

        print('sorting...')
        self.num_classes = 10
        self.class2labels = {} # n,
        self.class2data = {} # n, 3, h, w
        for i in range(self.num_classes):
            indices = np.where(labels==i)
            self.class2labels[i] = labels[indices]
            self.class2data[i] = np.transpose(data[indices], (0,2,3,1)) # (N,3,32,32) -> (N,32,32,3)
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
    nr_gpu = 8
    train_data = DataLoader('../../data', 'train', batch_size * nr_gpu, rng=None, shuffle=True, return_labels=True)
    print(train_data.next()[1])