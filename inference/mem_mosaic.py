#!/usr/bin/python3

# Work in progress, doing in memory mapping of tiles for additional performance gains
# https://towardsdatascience.com/efficiently-splitting-an-image-into-tiles-in-python-using-numpy-d1bf0dd7b6f7

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from time import time
from sys import argv


def _time(f):
    def wrapper(*args):
        start = time()
        r = f(*args)
        end = time()
        print("%s timed %f" % (f.__name__, end-start) )
        return r
    return wrapper


@_time
def for_loop_split(image: np.ndarray, kernel_size: tuple):

    img_height, img_width, channels = image.shape
    tile_height, tile_width = kernel_size

    tiled_array = np.zeros((img_height // tile_height,
                            img_width // tile_width,
                            tile_height,
                            tile_width,
                            channels))

    y = x = 0
    for i in range(0, img_height, tile_height):
        for j in range(0, img_width, tile_width):
            tiled_array[y][x] = image[i:i+tile_height,
                                      j:j+tile_width,
                                      :channels]
            x += 1
        y += 1
        x = 0

    return tiled_array


@_time
def stride_split(image: np.ndarray, kernel_size: tuple):
    # Image & Tile dimensions
    img_height, img_width, channels = image.shape
    tile_height, tile_width = kernel_size

    # bytelength of a single element
    bytelength = image.nbytes // image.size

    tiled_array = np.lib.stride_tricks.as_strided(
        image,
        shape=(img_height // tile_height,
               img_width // tile_width,
               tile_height,
               tile_width,
               channels),
        strides=(img_width*tile_height*bytelength*channels,
                 tile_width*bytelength*channels,
                 img_width*bytelength*channels,
                 bytelength*channels,
                 bytelength)
    )
    return tiled_array


@_time
def reshape_split(image: np.ndarray, kernel_size: tuple):

    img_height, img_width, channels = image.shape
    tile_height, tile_width = kernel_size

    tiled_array = image.reshape(img_height // tile_height,
                                tile_height,
                                img_width // tile_width,
                                tile_width,
                                channels)
    tiled_array = tiled_array.swapaxes(1, 2)
    return tiled_array


img = np.asarray(Image.open("captain-woof.jpg"))

t1, t2 = (argv[1], argv[2])
tilesize = (int(t1), int(t2))

tiles_1 = for_loop_split(img, tilesize)
tiles_2 = stride_split(img, tilesize)
tiles_3 = reshape_split(img, tilesize)

if (tiles_1 == tiles_2).all() and (tiles_2 == tiles_3).all():
    n = tiles_1.shape[0] * tiles_1.shape[1]
    print("\nAll tile arrays are equal.")
    print("Each array has %d tiles" % (n))
