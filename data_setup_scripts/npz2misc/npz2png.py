import numpy as np
from PIL import Image
import argparse

def npz2img(npz):
    "Gets the first array of images in an npz and returns the stack as an Image" 
    "Images are assumed to be HxWxC with values bw -1 and 1"
    all_arrs = np.load(npz)
    key = list(all_arrs.keys())[0] # first array name
    arr = all_arrs[key][:100]
    # arr = np.uint8((arr+1.)*float(255./2.)) # process
    arr = np.vstack(arr) # make one vertical image
    return Image.fromarray(arr)

def npz2png(npz:str, show:bool = False, outpath:str = None):
    img = npz2img(npz)
    if show: img.show()
    if outpath: img.save(outpath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Converts npz into png")
    parser.add_argument('npz', type=str,  help="path to npz file")
    parser.add_argument('-s', '--show', action='store_true', help="show image?")
    parser.add_argument('-o', '--out', type=str, help='optionally, where to save image')
    args = parser.parse_args()

    if not args.out:
        args.out = args.npz.replace('.npz', '.png')
    npz2png(args.npz, args.show, args.out)