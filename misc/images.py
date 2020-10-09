"Plotting images"

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def show(img: np.ndarray):
    if img.dtype == np.float:
        img = float_format_to_int_format(img)
    Image.fromarray(img).show()

def save(img: np.ndarray, img_path: str):
    if img.dtype == np.float:
        img = float_format_to_int_format(img)
    Image.fromarray(img).save(img_path)

def float_format_to_int_format(img: np.ndarray) -> np.ndarray:
    return ((img + 1.) * (255./2.)).astype(np.uint8)

def contrastify(ndar, eps=1e-8):
    """ Stretches all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar
    
# Plot image examples.
def save_with_title(img, title, path):
    plt.figure()
    plt.imshow(img, interpolation='none')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(path)
    plt.close('all')

def tile(imgs, aspect_ratio=1.0, grid_shape=None, border=1, border_color=3) -> np.array:
    ''' 
    Tile images in a grid.
    If grid_shape is provided, only as many images as specified in grid_shape
    will be included in the output.
    Otherwise aspect ratio will be used to pick the tile shape.
    Set border_color=0 for a white (invisible) border.
    '''
    
    # Choose grid shape
    n_imgs = len(imgs)
    img_shape = np.array(imgs.shape[1:3])
    if grid_shape is None:
        img_aspect_ratio = img_shape[1] / float(img_shape[0])
        aspect_ratio *= img_aspect_ratio
        tile_height = int(np.ceil(np.sqrt(n_imgs * aspect_ratio)))
        tile_width = int(np.ceil(np.sqrt(n_imgs / aspect_ratio)))
        grid_shape = np.array((tile_height, tile_width))
    else:
        assert len(grid_shape) == 2
        grid_shape = np.array(grid_shape)

    # Tile image shape
    tiled_img_shape = np.array(imgs.shape[1:])
    tiled_img_shape[:2] = (img_shape[:2] + border) * grid_shape[:2] - border

    # Assemble tile image
    tiled_img = np.empty(tiled_img_shape)
    tiled_img[:] = border_color
    for i in range(grid_shape[0]):
            for j in range(grid_shape[1]):
                    img_idx = j + i*grid_shape[1]
                    if img_idx >= n_imgs:
                            # No more images - stop filling out the grid.
                            break
                    img = imgs[img_idx]
                    yoff = (img_shape[0] + border) * i
                    xoff = (img_shape[1] + border) * j
                    tiled_img[yoff:yoff+img_shape[0], xoff:xoff+img_shape[1], ...] = img

    return tiled_img

def conv_filter_tile(filters):
    n_filters, n_channels, height, width = filters.shape
    grid_shape = None
    if n_channels == 3:
            # Interpret 3 color channels as RGB
            filters = np.transpose(filters, (0, 2, 3, 1))
    else:
            # Organize tile such that each row corresponds to a filter and the
            # columns are the filter channels
            grid_shape = (n_channels, n_filters)
            filters = np.transpose(filters, (1, 0, 2, 3))
            filters = np.resize(filters, (n_filters*n_channels, height, width))
    filters = contrastify(filters)
    return tile(filters, grid_shape=grid_shape)

