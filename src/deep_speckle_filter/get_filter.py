from keras import backend as K
import keras.models as models
from keras.layers.core import Layer, Reshape, Permute
import numpy as np
from filter_1 import getDecoder, getEncoder

kernel = 3
pad = 1
pool_size = 2


def getFilter(img_w=256, img_h=256, dim=3, model='1'):
    filter = models.Sequential()
    filter.add(Layer(input_shape=(img_h, img_w, dim)))
    for l in getEncoder():
        filter.add(l)
    for l in getDecoder():
        filter.add(l)
    filter.add(Reshape((1, img_h*img_w), input_shape=(1, img_h, img_w)))
    filter.add(Permute((2, 1)))
    return filter

if __name__ == '__main__':
    getFilter()