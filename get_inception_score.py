"""
Given a Tensorflow generative model, outputs its inception score
"""

import os
import json
import argparse

import numpy as np
from scoring import inception

def get_args():
    parser = argparse.ArgumentParser()
    # Oft-changed parameters
    parser.add_argument('-p', '--samples_path', type=str, default=None, help='The filepath to our samples')

    # Defaulted parameters
    parser.add_argument('-pd', '--preds_path', type=str, default=None, help='The filepath to our predictions, or where to save them')
    parser.add_argument('-np', '--num_predictions_', type=int, default=None, help='Num predictions to generate')
    parser.add_argument('-ns', '--num_splits_', type=int, default=1, help='Num splits for the inception score')
    parser.add_argument('-sdk', '--samples_data_key_', type=str, default='samples_np', help='Key of string for loading sample npz data')
    parser.add_argument('-pdk', '--preds_data_key_', type=str, default='preds', help='Key of string for loading prediction npz data')

    args = parser.parse_args()

    print('input args:\n', json.dumps(vars(args), indent=4, separators=(',',':'))) # pretty print args
    return args


if __name__ == "__main__":
    args = get_args()

    if not args.preds_path and not args.samples_path:
        print("must specify one of 'samples_path' (-p) or 'preds_path' (-pd)")
        os.exit(1)

    if args.preds_path:
        preds_path = args.preds_path
    else:
        # https://stackoverflow.com/questions/2556108/rreplace-how-to-replace-the-last-occurrence-of-an-expression-in-a-string
        def rreplace(s, old, new, occurrence):
            li = s.rsplit(old, occurrence)
            return new.join(li)

        preds_path = rreplace(args.samples_path, 'samples', 'preds', 1)

    print('preds_path: {}'.format(preds_path))
    if os.path.exists(preds_path):
        print('loading predictions from {}...'.format(preds_path))
        preds = np.load(preds_path)[args.preds_data_key_]
    else:
        if not os.path.exists(args.samples_path):
            print('samples path [{}] does not exist, exiting.'.format(args.samples_path))
            exit(1)
        print('loading samples from {}...'.format(args.samples_path))
        samples = np.load(args.samples_path)[args.samples_data_key_]
        samples = list(samples)
        if args.num_predictions_:
            samples = samples[:args.num_predictions]
        process = lambda img: ((img + 1) * 255 / 2).astype('uint8')
        samples = [process(s) for s in samples]
        print('getting predictions on {} samples...'.format(len(samples)))
        preds = inception.get_inception_preds(samples)
        print('saving predictions to {} ...'.format(preds_path))
        np.savez(preds_path, preds=preds)

    if args.num_predictions_:
        preds = preds[:args.num_predictions]
    print('getting inception score on {} predictions with {} splits...'.format(len(list(preds)), args.num_splits_))
    mean, var = inception.get_inception_score_from_preds(preds, splits=args.num_splits_)
    print('inception score: mean={}, variance={}'.format(mean, var))
