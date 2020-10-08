"Create unconditional dataset of simple images with yellow circles and purple squares."

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
    for i in range(n):
        img = Image.new('RGB', (int(image_dim), int(image_dim)))
        drawer = ImageDraw.Draw(img)

        circle = random.randint(0,1) == 0 # coin flip
        # draw
        if circle:
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
    return np.stack(imgs) # put into a single numpy array w shape (N, d, d, c)

def download_if_not_downloaded(shapes_dir, subset):
    if not os.path.exists(shapes_dir):
        os.makedirs(shapes_dir)
        print(f'created {shapes_dir}.')
    filename = f'{subset}.npy'
    filepath = os.path.join(shapes_dir, filename)
    if not os.path.exists(filepath):
        subset2samples = {'train': 60000, 'test': 10000}
        x_arr = create_shapes_dataset(n=subset2samples[subset])
        np.save(filepath, x_arr)
    return filename

def load_downloaded_shapes(shapes_dir, filename):
    "loads as numpy arrays: xs as shape (N, 32, 32, 3) and ys as shape (N,)"
    filepath = os.path.join(shapes_dir, filename)
    x = np.load(filepath)
    return x, None

def load_shapes(data_dir, subset='train'):
    assert(subset in ('train', 'test'))
    shapes_dir = os.path.join(data_dir, 'shapes')
    filename = download_if_not_downloaded(shapes_dir, subset)
    x, y = load_downloaded_shapes(shapes_dir, filename)
    return x, y # trainx should be (N,32,32,3), trainy should be (N,)

class DataLoader(object):
    """ an object that generates batches of shapes data for training """

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

        # create temporary storage for the data, if not yet created
        if not os.path.exists(data_dir):
            print('creating folder', data_dir)
            os.makedirs(data_dir)

        # load shapes training data to RAM
        self.data, self.labels = load_shapes(data_dir, subset=subset)
        #np.savez('../../data/SVHNz', trainx=self.data)
        
        self.p = 0 # pointer to where we are in iteration
        self.rng = np.random.RandomState(1) if rng is None else rng

    def get_observation_size(self):
        return self.data.shape[1:]

    def get_num_labels(self):
        return np.amax(self.labels) + 1

    def reset(self):
        self.p = 0

    def __iter__(self):
        return self

    def __next__(self, n=None):
        """ n is the number of examples to fetch """
        if n is None: n = self.batch_size

        # on first iteration lazily permute all data
        if self.p == 0 and self.shuffle:
            inds = self.rng.permutation(self.data.shape[0])
            self.data = self.data[inds]
            if self.return_labels: self.labels = self.labels[inds]

        # on last iteration reset the counter and raise StopIteration
        if self.p + n > self.data.shape[0]:
            self.reset() # reset for next time we get called
            raise StopIteration

        # on intermediate iterations fetch the next batch
        x = self.data[self.p : self.p + n]
        if self.return_labels: y = self.labels[self.p : self.p + n]
        self.p += self.batch_size

        if self.return_labels:
            return x,y
        else:
            return x

    next = __next__  # Python 2 compatibility (https://stackoverflow.com/questions/29578469/how-to-make-an-object-both-a-python2-and-python3-iterator)

# Testing data loader
if __name__ == "__main__":
    batch_size = 16
    nr_gpu = 8
    train_data = DataLoader('../../data', 'train', batch_size * nr_gpu, rng=None, shuffle=True, return_labels=False)
    imgs = train_data.next()

    # show img
    from PIL import Image
    # img = Image.fromarray(imgs[0]) # from numpy array to image
    # img.show()
    for i in range(10):
        Image.fromarray(imgs[i]).show()