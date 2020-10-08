from __future__ import print_function
import argparse
import os
import scipy.io as sio
import numpy as np

# OLD VERSION.
# def mat2npz(mat_path, train_split, out_path):
#     "Processes images in mat and saves them to npz file."
#     dim = 28
#     data = sio.loadmat(mat_path)
#     X, y = data['X'], data['y']
#     data = arr[0].reshape((-1,28,28,1)).transpose(0,2,1,3)
#     labels = arr[1].reshape((-1)) - 1 # switch to zero indexing
#     num = len(labels)
#     np.savez(out_path, trainx=data[:int(train_split * num)], testx=data[int(train_split * num):],
#                         trainy=labels[:int(train_split * num)], testy=labels[int(train_split * num):])

# NEW VERSION. Works with SVHN .mat files
def mat2npz(mat_path, train_split, out_path):
    "Processes images in mat and saves them to npz file."
    dim = 28
    data = sio.loadmat(mat_path)
    X, y = data['X'].transpose(3,0,1,2), data['y'].reshape((-1))
    if np.min(y) == 1: y = y-1 # switch to zero indexing
    num = len(y)
    np.savez(out_path, trainx=X[:int(train_split * num)], testx=X[int(train_split * num):],
                        trainy=y[:int(train_split * num)], testy=y[int(train_split * num):])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('inpath', type=str, default=None, help='Location of the .mat file')
    parser.add_argument('-s', '--split', type=float, default=1, help='Train-test split (a value between 0 and 1)' )
    parser.add_argument('-o', '--outpath', type=str, default=None, help='Location for generated .npz file' )    
    args = parser.parse_args()

    if args.outpath == None:
        path, ext = os.path.splitext(args.inpath) # e.g. 'save/file', '.ext'
        args.outpath = f'{path}.npz'  

    mat2npz(args.inpath, args.split, args.outpath)