from __future__ import print_function
import argparse
import cv2
import lmdb
import numpy as np
import os
from os.path import exists, join
from skimage.transform import resize
import sys

def crop_square(arr):
    "Center crops array to be square"
    h = w = min(arr.shape[:2])
    return crop(arr, h, w)

def crop(arr, h, w):
    "Center crops array to have height h and width w"
    y,x = arr.shape[0]//2, arr.shape[1]//2 # center
    return arr[y-h//2:y+h//2, x-w//2:x+w//2]

def dir2npz(db_path, train_split, out_path, dim=32):
    "Processes images in db dir (center crops to dim x dim) and saves them to npz file."
    env = lmdb.open(db_path, map_size=1099511627776, max_readers=100, readonly=True)
    with env.begin(write=False) as txn:
        arrs = []
        cursor = txn.cursor()
        i = 0
        for key, val in cursor:
            arr = cv2.imdecode(np.fromstring(val, dtype=np.uint8), 1)
            if len(arr.shape) == 3 and arr.shape[2] == 3 and arr.shape[0] > dim and arr.shape[1] > dim:  # drop badly shaped images and too small images
                arr = crop_square(arr)  # make square
                arr = (resize(arr, (dim, dim)) * 255).astype('uint8')  # make dim x dim
                assert (arr.shape == (dim, dim, 3))
                arrs.append(arr)
                # Image.fromarray(arr).show()
                # print(arr.shape)
                # n += 1
                # print(n)

                i += 1
                if i % 1000 == 0:
                    print('.', end='', flush=True)
                    if i == 30000:
                        np.savez(out_path+str('_30k'), trainx=arrs[:-1], testx=arrs[-1:])
        np.savez(out_path, trainx=arrs[:int(train_split * len(arrs))], testx=arrs[int(train_split * len(arrs)):])



if __name__ == "__main__":
    args = sys.argv[1:]
    db_path = args[0] # path to database directory
    train_split = float(args[1]) if len(args)>=2 else 1
    out_path = args[2] if len(args)>=3 else db_path+'.npz'

    dir2npz(db_path, train_split, out_path)