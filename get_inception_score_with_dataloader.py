"""
Given a Tensorflow generative model, outputs its inception score
"""

import os
import json
import argparse
import random

import numpy as np
from scoring import inception

def get_args():
    parser = argparse.ArgumentParser()
    # Oft-changed parameters
    parser.add_argument('-d', '--data_set', type=str, default='cifar', help='Can be either cifar|imagenet')
    parser.add_argument('-pd', '--preds_path', required=True, type=str, default=None, help='The filepath to our predictions, or where to save them')

    # Defaulted parameters
    parser.add_argument('-i', '--data_dir', type=str, default='../data', help='Location for the dataset')
    parser.add_argument('-np', '--num_predictions_', type=int, default=None, help='Num predictions to generate')
    parser.add_argument('-ns', '--num_splits_', type=int, default=1, help='Num splits for the inception score')

    args = parser.parse_args()

    print('input args:\n', json.dumps(vars(args), indent=4, separators=(',',':'))) # pretty print args
    return args


if __name__ == "__main__":
    args = get_args()

    preds_path = args.preds_path
    if os.path.exists(preds_path):
        print('loading predictions from {}...'.format(preds_path))
        preds = np.load(preds_path)['preds']
    else:
    if args.data_set == 'imagenet':
        import data.imagenet_data as imagenet_data
        DataLoader = imagenet_data.DataLoader
    elif args.data_set == 'cifar':
        import data.cifar10_data as cifar10_data
        DataLoader = cifar10_data.DataLoader
    else:
        print('data_set (-d) must be either cifar|imagenet')
        exit(1)

    print('loading samples from {}|{}...'.format(args.data_dir, args.data_set))
    train_data = DataLoader(args.data_dir, 'train', 100, shuffle=False, return_labels=False)
    samples = train_data.data
    samples = list(samples)
    random.shuffle(samples)
    if args.num_predictions_:
        samples = samples[:args.num_predictions_]

    print(np.min(samples[0]))
    print(np.max(samples[0]))

    # process = lambda img: ((img + 1) * 255 / 2).astype('uint8')
    # samples = [process(s) for s in samples]

    print('getting predictions on {} samples...'.format(len(samples)))
    preds = inception.get_inception_preds(samples)
    print('saving predictions to {} ...'.format(preds_path))
    np.savez(preds_path, preds=preds)

    if args.num_predictions_:
        preds = preds[:args.num_predictions_]
    print('getting inception score on {} predictions with {} splits...'.format(len(list(preds)), args.num_splits_))
    mean, var = inception.get_inception_score_from_preds(preds, splits=args.num_splits_)
    print('inception score: mean={}, variance={}'.format(mean, var))
