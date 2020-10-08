# import requests
# from pathlib import Path
import argparse
import os
import sys
from skimage import io
import numpy as np
from skimage.transform import resize

sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..'))) # Adds higher directory to python modules path.
import data_setup_util as data_util

dataset_name = 'omniglot'

def download_omniglot_if_necessary(data_path, force=False):
    """If omniglot does not exist at save_path, download it to save_path)"""
    train_url = 'https://raw.github.com/brendenlake/omniglot/master/python/images_background.zip'
    test_url = 'https://raw.github.com/brendenlake/omniglot/master/python/images_evaluation.zip'
    train_zip_path, test_zip_path = data_util.download_dataset_if_necessary(data_path, dataset_name, train_url, test_url)
    return train_zip_path, test_zip_path

def preprocess_omniglot_img(img):
    # print(img.dtype, img.shape, np.min(img), np.max(img), np.median(img))
    # io.imshow(img)
    # io.show()
    img = resize(img, (28,28,3), preserve_range=True, anti_aliasing=True).astype(np.uint8)
    # print(img.dtype, img.shape, np.min(img), np.max(img), np.median(img))
    # io.imshow(img)
    # io.show()
    return img

def omniglot2npz(data_path, train_zip_path, test_zip_path):
    dataset_path = os.path.join(data_path, dataset_name)
    extracted_path = os.path.join(dataset_path, 'extracted')
    data_util.extract_zip(train_zip_path, extracted_path), data_util.extract_zip(test_zip_path, extracted_path)
    subset_name2path = {'trainx': os.path.join(extracted_path, 'images_background'),
                        'testx': os.path.join(extracted_path, 'images_evaluation')}
    subset_name2subset = {}
    for subset_name, subset_path in subset_name2path.items():
        subset = []
        for alphabet_name in os.listdir(subset_path):
            alphabet_path = os.path.join(subset_path, alphabet_name)
            for character_name in os.listdir(alphabet_path):
                character_path = os.path.join(alphabet_path, character_name)
                for png_name in os.listdir(character_path):
                    png_path = os.path.join(character_path, png_name)
                    img = io.imread(png_path)
                    subset.append(preprocess_omniglot_img(img))
        # end of subset
        subset_name2subset[subset_name] = np.stack(subset) # shape: N_IMAGES x W x H (x C)
    np.savez( f'{dataset_path}.npz', **subset_name2subset)
    # done with both subsets

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', type=str, help='Path to local data directory')
    # parser.add_argument('-o', '--out_path', type=str, help='Path to directory to put npz files')
    args = parser.parse_args()
    train_zip_path, test_zip_path = download_omniglot_if_necessary(args.data_path)
    omniglot2npz(args.data_path, train_zip_path, test_zip_path)