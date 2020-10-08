import argparse
import numpy as np

def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)
        images = images.reshape(-1, 28, 28, 1)


    return images, labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--dir_path', type=str, help='Location of the dir containing the gzip files')
    args = parser.parse_args()

    trainx, trainy = load_mnist(path=args.dir_path, kind='train')
    testx, testy = load_mnist(path=args.dir_path, kind='t10k')
    np.savez('../fashion.npz', trainx=trainx, trainy=trainy, testx=testx, testy=testy)