import tensorflow as tf
import tensorlayer as tl

import scipy, random
import numpy as np
from functools import reduce

def get_gray_imgs_fn(file_name, path):
    """ Input an image path and name, return an image array """
    # return scipy.misc.imread(path + file_name).astype(np.float)
    return scipy.misc.imread(path + file_name, mode='L')

def save_image(image, path):
    """Save an image as a png file."""
    min_val = image.min()
    if min_val < 0:
        image = image + min_val

    # image = (image.squeeze() * 1.0 / image.max()) * 255
    # image = image.astype(np.uint8)

    scipy.misc.imsave(path, image)
    print('[#] Image saved {}.'.format(path))



def norm(x):
    x = x / (255. / 2.)
    x = x - 1.
    # x = x / 255
    return x


def inv_norm(x):
    x = x + 1
    x = x * (255. / 2.)
    return x

def inv_norm_0(x):
    x = x - min(x)
    x = x / max(x)
    x = x*255
    return x

def augm(x):
    size = x.shape
    x = tl.prepro.flip_axis(x, axis=0, is_random=True)
    x = tl.prepro.flip_axis(x, axis=1, is_random=True)
    x = np.reshape(x, (size[0], size[1], 1))
    rg = random.sample([0, 90, 180, 270], 1)
    rg = rg[0]
    x = tl.prepro.rotation(x, rg=rg, is_random=False)
    return x


def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)

def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10,dtype=numerator.dtype))
    return numerator / denominator