import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--data_path', type=str, help='Location of the npz')
    parser.add_argument('-o', '--out_path', type=str, help='Out path')
    args = parser.parse_args()

    loaded = np.load(args.data_path)
    assert len(loaded['trainx'].shape) == 4, 'Shape should be BxHxWx1 but had only 3 dimensions.'
    trainx = np.repeat(loaded['trainx'],3,axis=3)
    testx = np.repeat(loaded['testx'],3,axis=3)
    np.savez(args.out_path, trainx = trainx, testx = testx, trainy = loaded['trainy'], testy = loaded['testy'])