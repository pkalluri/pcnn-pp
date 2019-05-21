"For each sample, find the nearest image pixel-wise in the training set."

import numpy as np
from PIL import Image

def process(img):
    return np.uint8((img + 1)*255/2)

def find_nearest(samples,dataset):
    '''
    :param samples: NxHxWx3
    :param dataset: NxHxWx3
    :return:
    '''
    imgs = []
    neighbors = []
    for img in samples:
        img = process(img) # 0 to 255
        imgs.append(img)

        pixel_distances = (dataset - img.reshape((1,32,32,3)))**2
        image_distances = np.sum(pixel_distances, axis=(1,2,3))
        nearest_index = np.argmin(image_distances)
        nearest = dataset[nearest_index]
        neighbors.append(nearest)

    grid = np.hstack([np.vstack(imgs),np.vstack(neighbors)])
    Image.fromarray(grid).show()



# some files to have handy
fox_sum = '703871/imagenet_small_sample720.npz'
fox_data = 'red-fox.npz'

soccer_ball_max = '706919_soccer-ball_log_max/soccer-ball_sample540.npz'
soccer_st = '706918_soccer-ball_standard/soccer-ball_sample280.npz'
soccer_data = 'soccer-ball.npz'

sun_max = '705140/imagenet_sunflower_small_sample1740.npz'
sun_sum = '705134/imagenet_sunflower_small_sample2160.npz'
sun_sum_early = '705134/imagenet_sunflower_small_sample300.npz'
sun_st = '705136/imagenet_sunflower_small_sample740.npz'
sun_data = 'sunflower.npz'



# Set args
samples = 'save/'+fox_sum
data = '../data/'+fox_data



find_nearest(np.load(samples)['arr_0'],
             np.load(data)['trainx'])