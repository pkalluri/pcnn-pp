from PIL import Image
import numpy as np
import os
import re
import sys
import argparse


def zero_centered2rgb(arr):
    assert(np.all(arr>=-1) and np.all(arr<=1))
    return ((arr + 1) * (255/2)).astype(np.uint8)

def save_all_png(arr, out_path):
    imgs = arr
    N_EPOCHS, BATCHSIZE, H, W, C = imgs.shape
    imgs = imgs.swapaxes(1,2) # shape: N_EPOCHS x H x BATCHSIZE x W x C
    imgs = imgs.reshape((N_EPOCHS*H, BATCHSIZE*W, C))
    Image.fromarray(imgs).save(out_path)

def save_final_png(arr, out_path):
    imgs = arr
    N_EPOCHS, BATCHSIZE, H, W, C = imgs.shape
    imgs = imgs.swapaxes(1,2) # shape: N_EPOCHS x H x BATCHSIZE x W x C
    imgs = imgs.reshape((N_EPOCHS*H, BATCHSIZE*W, C))    
    final_img = imgs[-H:,:,:]
    Image.fromarray(final_img).save(os.path.join(dir_path,'final_samples.png'))

def condense_samples(dir_path, pattern='.*_sample.*\..npz'):
    npz_paths = [os.path.join(dir_path,fn) for fn in os.listdir(dir_path) if  fn.endswith('.npz') and re.match(pattern)] # get all npzs

    if npz_paths:
        npz_paths.sort(key=lambda path: int(re.findall(r'\d+', path)[-1])) # sort in ascending order
        imgs = zero_centered2rgb(np.stack([np.load(npz_path)['arr_0'] for npz_path in npz_paths])) # consolidate
        
        # save only one big arr
        np.save(os.path.join(dir_path,'training_samples'), imgs) # save; shape: N_EPOCHS x BATCHSIZE x H x W x C
        # save one big img
        save_all_png(imgs, os.path.join(dir_path,'training_samples.png'))

        # save one small img representing the end of training
        save_final_png(imgs, os.path.join(dir_path,'final_samples.png'))

        # rm original npz and pngs
        removing = [os.remove(os.path.join(dir_path,fn)) for fn in os.listdir(dir_path) if re.match('.*_sample\d+\.[png | npz]', fn)]

def condense_dir(dir_path):
    condense_samples(dir_path)

    # TO BE DELETED
    npz_paths = [os.path.join(dir_path,fn) for fn in os.listdir(dir_path) if  fn.endswith('.npz') and not fn.startswith('test_bpd')] # get all npzs
    
    # os.remove(os.path.abspath(os.path.join(dir_path,'test_bpd'))) # trash file
    if len(npz_paths) > 0:
        npz_paths.sort(key=lambda path: int(re.findall(r'\d+', path)[-1])) # sort in ascending order
        imgs = zero_centered2rgb(np.stack([np.load(npz_path)['arr_0'] for npz_path in npz_paths])) # consolidate
        
        # save only one big arr
        np.save(os.path.join(dir_path,'training_samples'), imgs) # save; shape: N_EPOCHS x BATCHSIZE x H x W x C
        # save one big img
        save_all_png(imgs, os.path.join(dir_path,'training_samples.png'))

        # save one small img representing the end of training
        save_final_png(imgs, os.path.join(dir_path,'final_samples.png'))

        # rm original npz and pngs
        removing = [os.remove(os.path.join(dir_path,fn)) for fn in os.listdir(dir_path) if re.match('.*_sample\d+\.[png | npz]', fn)]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('meta_dir_path', type=str, help='Directory path. All subdirectories match pattern (see below) will be condensed.')
    parser.add_argument('pattern', type=str, default='\d+_.*', help='All subdirectories that match this regex pattern will be condensed.')
    for name in os.listdir(args.meta_dir_path):
        path = os.path.join(args.meta_dir_path, name)
        if os.path.isdir(path) and re.match(args.pattern, path):
            condense_dir(path)
