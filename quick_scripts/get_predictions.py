"Given samples, get and save InceptionNet's predicted class probabilities."

import argparse
import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..'))) # Adds higher directory to python modules path.
from evaluation import inception

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('samples_path', type=str, help='Path to samples (npz)')
    parser.add_argument('preds_path', type=str, help='Path to put generated predictions (npz)')
    args = parser.parse_args()

    samples = np.load(args.samples_path)
    preds = inception.samples_to_predictions(samples)
    np.save(args.preds_path, preds)
