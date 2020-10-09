"""
Converts directory of images into a single npz file.
"""
from PIL import Image
import numpy as np
import os
from skimage.transform import resize

def crop_square(arr):
    "Center crops array to be square"
    h = w = min(arr.shape[:2])
    return crop(arr, h, w)

def crop(arr, h, w):
    "Center crops array to have height h and width w"
    y,x = arr.shape[0]//2, arr.shape[1]//2 # center
    return arr[y-h//2:y+h//2, x-w//2:x+w//2]

def dir2npz(dir, dim=32, train_split=.9):
    "Processes images in dir (crops to dim x dim) and saves processed images to dir.npz"
    arrs = []
    n = 0
    for f in os.listdir(dir):
        fp = os.path.join(dir,f)
        try:
            arr = np.asarray(Image.open(fp))
            if len(arr.shape)==3 and arr.shape[2]==3 and arr.shape[0]>dim and arr.shape[1]>dim: # drop badly shaped images and too small images
                arr = crop_square(arr) # make square
                arr = (resize(arr, (dim, dim))*255).astype('uint8') # make dim x dim
                assert(arr.shape == (dim,dim,3))
                arrs.append(arr)
                # Image.fromarray(arr).show()
                # print(arr.shape)
                # n += 1
                # print(n)
        except UnboundLocalError:
            print('couldnt read')
    np.savez(dir+'.npz', trainx = arrs[:int(train_split*len(arrs))], testx = arrs[int(train_split*len(arrs)):])

dir2npz('soccer-ball')
print('done')