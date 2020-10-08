"Create conditional dataset of simple images with red circles or blue squares."

from PIL import Image, ImageDraw
import random
import numpy as np
import os

def concat_horizontal(im1, im2):
    img = Image.new('RGB', (im1.width + im2.width, im1.height))
    img.paste(im1, (0, 0))
    img.paste(im2, (im1.width, 0))
    return img

def create_shapes_dataset(n, image_dim=32, min_shape_size=8, max_shape_size=16, res=100):
    """
    n: number of images
    image_dim: dimension (height and width) of images
    min(max)_shape_size: min(max) number of pixels for the width of the shape
    res: shapes are drawn at a resolution of res * image_dim
    """
    imgs = []
    labels = []
    for i in range(n):
        img = Image.new('RGB', (int(image_dim), int(image_dim)))
        drawer = ImageDraw.Draw(img)

        draw_circle = random.randint(0,1) == 0 # coin flip
        labels.append(int(draw_circle))
        # draw
        if draw_circle:
            diam = random.randint(min_shape_size, max_shape_size) # diameter
            # diam = 16 # dbg
            loc_x, loc_y = random.randint(0, image_dim-diam), random.randint(0, image_dim-diam) # loc in image
            color = (255, 0, 0)
            drawer.ellipse( (loc_x, loc_y, loc_x+diam, loc_y+diam), fill=color)
            # print((loc_x, loc_y, loc_x+diam, loc_y+diam), color) # dbg
        else: # square
            side = random.randint(min_shape_size, max_shape_size) # i.e. height and width
            loc_x, loc_y = random.randint(0, image_dim-side), random.randint(0, image_dim-side) # loc in image
            color = (0,0,255) #blue
            drawer.rectangle( (loc_x, loc_y, loc_x+side, loc_y+side), fill=color)
            # print((loc_x, loc_y, loc_x+side, loc_y+side), color) # dbg
        imgs.append(np.array(img))
    # done making images
    return np.stack(imgs), np.array(labels) # xs have shape (N, d, d, c), ys have shape (N,)

def download_if_not_downloaded(shapes_dir, subset):
    if not os.path.exists(shapes_dir):
        os.makedirs(shapes_dir)
        print(f'created {shapes_dir}.')
    filename = f'{subset}.npz'
    
    # create
    filepath = os.path.join(shapes_dir, filename)
    if not os.path.exists(filepath):
        subset2samples = {'train': 60000, 'test': 10000}
        x,y = create_shapes_dataset(n=subset2samples[subset])
        np.savez(filepath, x, y)

    return filename

def load_downloaded_shapes(shapes_dir, filename):
    "loads as numpy arrays: xs as shape (N, 32, 32, 3) and ys as shape (N,)"
    filepath = os.path.join(shapes_dir, filename)
    arrays = np.load(filepath)
    x, y = arrays['arr_0'], arrays['arr_1']
    return x, y

def load_shapes(data_dir, subset='train'):
    assert(subset in ('train', 'test'))
    shapes_dir = os.path.join(data_dir, 'shapes')
    filename = download_if_not_downloaded(shapes_dir, subset)
    x, y = load_downloaded_shapes(shapes_dir, filename)
    return x, y # trainx should be (N,32,32,3), trainy should be (N,)

class DataLoader(object):
    """ an object that generates batches of SVHN data for training """

    def __init__(self, data_dir, subset, batch_size, rng=None, shuffle=False, return_labels=False):
        """
        - data_dir is location where to store files
        - subset is train|test
        - batch_size is int, of #examples to load at once
        - rng is np.random.RandomState object for reproducibility
        """

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.return_labels = return_labels

        # create temporary storage for data, if not yet created
        if not os.path.exists(data_dir):
            print('creating folder', data_dir)
            os.makedirs(data_dir)
        print('loading...')
        # load SVHN training data to RAM. data should be (N,32,32,3), labels should be (N,)
        data, labels = load_shapes(data_dir, subset=subset)

        print('sorting...')
        self.num_classes = 2
        self.class2labels = {} # n,
        self.class2data = {} # n, 3, h, w
        for i in range(self.num_classes):
            indices = np.where(labels==i)
            self.class2labels[i] = labels[indices]
            self.class2data[i] = data[indices] # (N,32,32,3)
            # np.savez('cifar-class{}'.format(i), trainx=self.class2data[i], trainy = self.class2labels[i])

        self.p = 0 # pointer to where we are in iteration
        self.rng = np.random.RandomState(1) if rng is None else rng

        self.batches_labels = []
        self.batches_data = []
        self.reset(force_shuffle=True)
        # print('batch shape', self.batches_data[0].shape, self.batches_labels[0].shape)

    def get_observation_size(self):
        # return self.data.shape[1:]
        # print(self.class2data[0].shape[1:])
        return self.class2data[0].shape[1:]

    def get_num_labels(self):
        # return np.amax(self.labels) + 1
        return 10

    def reset(self, force_shuffle=False):
        self.p = 0
        # lazily permute all data
        if self.shuffle or force_shuffle:
            # create new batches
            # print('fresh batches...')
            self.batches_labels = []
            self.batches_data = []
            for i in range(self.num_classes):
                n = len(self.class2labels[i])
                inds = self.rng.permutation(n)
                i_labels_shuffled = self.class2labels[i][inds]
                i_data_shuffled = self.class2data[i][inds]
                for batch_start in list(range(0,n,self.batch_size))[:-1]:
                    self.batches_labels.append(i_labels_shuffled[batch_start:batch_start+self.batch_size])
                    self.batches_data.append(i_data_shuffled[batch_start:batch_start+self.batch_size])

            # shuffle batches
            # print('shuffling...')
            combined = list(zip(self.batches_labels, self.batches_data))
            random.shuffle(combined)
            self.batches_labels[:], self.batches_data[:] = zip(*combined)

    def __iter__(self):
        return self

    def __next__(self, n=None):
        """ n is the number of examples to fetch """
        assert(n is None or n % self.batch_size==0)
        if n is None:
            n = self.batch_size

        num_batches = n // self.batch_size

        # on last iteration reset the counter and raise StopIteration
        if self.p + num_batches - 1 >= len(self.batches_labels):
            raise StopIteration

        # on intermediate iterations fetch the next batch
        x = self.batches_data[self.p:self.p + num_batches]
        x = np.concatenate(x)
        y = self.batches_labels[self.p:self.p + num_batches]
        y = np.concatenate(y)
        self.p += num_batches

        if self.return_labels:
            return x,y
        else:
            return x

    next = __next__  # Python 2 compatibility (https://stackoverflow.com/questions/29578469/how-to-make-an-object-both-a-python2-and-python3-iterator)

# Testing data loader
if __name__ == "__main__":
    batch_size = 16
    nr_gpu = 8
    train_data = DataLoader('../../data', 'train', batch_size * nr_gpu, rng=None, shuffle=True, return_labels=True)
    imgs, labels = train_data.next()
    print(labels)

    # show img
    from PIL import Image
    for i in range(10):
        Image.fromarray(imgs[i]).show()