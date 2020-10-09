import requests
# from pathlib import Path
import zipfile
import os

def download_dataset_if_necessary(data_path, dataset_name, train_url, test_url, force_download=False):
    dataset_path = os.path.join(data_path,dataset_name)
    os.makedirs(dataset_path, exist_ok = True)
    subset2url = {'train': train_url,
                  'test': test_url}
    fnames = []
    subset_paths = []
    for (subset, url) in subset2url.items(): # train and test subsets
        _, ext = os.path.splitext(url)
        subset_path = os.path.join(dataset_path, f'{subset}{ext}') # e.g. train.zip
        subset_paths.append(subset_path)
        if not os.path.exists(subset_path) or force_download: # download
            zip_content = requests.get(url).content
            with open(subset_path,'wb') as f:
                f.write(zip_content)
    return subset_paths # train file and test file

def extract_zip(zip_path, out_dir_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_file:
        zip_file.extractall(out_dir_path)