"For each sample, find the nearest image pixel-wise in the training set."

import numpy as np
from PIL import Image
import argparse

def process(img):
    return np.uint8((img + 1)*255/2)

def find_nearest(samples,dataset):
    '''
    :param samples: NxHxWx3
    :param dataset: NxHxWx3
    '''
    imgs = []
    neighbors = []
    for s in samples:
        img = process(s) # 0 to 255
        imgs.append(img)

        pixel_distances = (dataset - img.reshape((1,32,32,3)))**2
        image_distances = np.sum(pixel_distances, axis=(1,2,3))
        nearest_index = np.argmin(image_distances)
        nearest_img = dataset[nearest_index]
        neighbors.append(nearest_img)

    grid = np.hstack([np.vstack(imgs),np.vstack(neighbors)])
    summary_image = Image.fromarray(grid)
    summary_image.show()
    return summary_image


# # some files to have handy
# fox_sum = '703871/imagenet_small_sample720.npz'
# fox_data = 'red-fox.npz'
#
# soccer_ball_max = '706919_soccer-ball_log_max/soccer-ball_sample540.npz'
# soccer_st = '706918_soccer-ball_standard/soccer-ball_sample280.npz'
# soccer_data = 'soccer-ball.npz'
#
# sun_max = '705140/imagenet_sunflower_small_sample1740.npz'
# sun_sum = '705134/imagenet_sunflower_small_sample2160.npz'
# sun_sum_early = '705134/imagenet_sunflower_small_sample300.npz'
# sun_st = '705136/imagenet_sunflower_small_sample740.npz'
# sun_data = 'sunflower.npz'
#
# # Set args
# samples = 'save/'+fox_sum
# data = '../data/'+fox_data
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--generated', type=str, help='Location of the generated samples npz')
    parser.add_argument('-d', '--data', type=str, help='Location of the training data npz')
    parser.add_argument('-o', '--out', type=str, help='Location to put the summary image')
    args = parser.parse_args()

    summary_image = find_nearest(np.load(args.generated)['samples_np'],np.load(args.data)['trainx'])
    summary_image.save(args.generated.replace('.npz', '_nearest_neighbors.png'))
