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

def cifar2sortednpz(data_dir, subset='train', num_classes=10):
    print('loading...')
    # load CIFAR-10 training data to RAM
    data, labels = load(os.path.join(data_dir,'cifar-10-python'), subset=subset)
    data = np.transpose(data, (0,2,3,1)) # (N,3,32,32) -> (N,32,32,3)

    class2labels = {} # n,
    class2data = {} # n, 3, h, w
    for i in range(num_classes):
        indices = np.where(labels==i)
        i_labels = labels[indices]
        i_data = np.transpose(data[indices], (0,2,3,1)) # (N,3,32,32) -> (N,32,32,3)
    #
    # self.p = 0 # pointer to where we are in iteration
    # self.rng = np.random.RandomState(1) if rng is None else rng

    # print('creating new batches...')
    # # create new batches
    # self.batches_labels = []
    # self.batches_data = []
    # for i in range(self.num_classes):
    #     n = len(self.class2labels[i])
    #     inds = self.rng.permutation(n)
    #     i_labels_shuffled = self.class2labels[i][inds]
    #     i_data_shuffled = self.class2data[i][inds]
    #     for batch_start in list(range(0,n,batch_size))[:-1]:
    #         self.batches_labels.append(i_labels_shuffled[batch_start:batch_start+batch_size])
    #         self.batches_data.append(i_data_shuffled[batch_start:batch_start+batch_size])

    # print('shuffling batches...')
    # # shuffle batches
    # combined = list(zip(self.batches_labels, self.batches_data))
    # random.shuffle(combined)
    # self.batches_labels[:], self.batches_data[:] = zip(*combined)

# Testing sorted data loader
if __name__ == "__main__":
    cifar2sortednpz('../../data')