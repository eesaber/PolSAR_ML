from __future__ import absolute_import
from __future__ import print_function
import os


import keras.models as models
from keras.layers.core import Layer, Dense, Dropout, Flatten, Reshape, Merge, Permute
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization


from keras import backend as K
import numpy as np
np.random.seed(7) # 0bserver07 for reproducibility


img_w = 1024
img_h = 1024
n_labels = 12

kernel = 3
pad = 1
pool_size = 2
def getEncoder():
    encoding_layers = [
        Conv2D(64, kernel, kernel, activation='relu', border_mode='same'),
        BatchNormalization(),
        Conv2D(64, kernel, kernel, activation='relu', border_mode='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        Conv2D(128, kernel, kernel, activation='relu', border_mode='same'),
        BatchNormalization(),
        Conv2D(128, kernel, kernel, activation='relu', border_mode='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        Conv2D(256, kernel, kernel, activation='relu', border_mode='same'),
        BatchNormalization(),
        Conv2D(256, kernel, kernel, activation='relu', border_mode='same'),
        BatchNormalization(),
        Conv2D(256, kernel, kernel, activation='relu', border_mode='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        Conv2D(512, kernel, kernel, activation='relu', border_mode='same'),
        BatchNormalization(),
        Conv2D(512, kernel, kernel, activation='relu', border_mode='same'),
        BatchNormalization(),
        Conv2D(512, kernel, kernel, activation='relu', border_mode='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        Conv2D(512, kernel, kernel, activation='relu', border_mode='same'),
        BatchNormalization(),
        Conv2D(512, kernel, kernel, activation='relu', border_mode='same'),
        BatchNormalization(),
        Conv2D(512, kernel, kernel, activation='relu', border_mode='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(pool_size, pool_size)),
    ]
    return encoding_layers

def getDecoder():
    decoding_layers = [
        UpSampling2D(size=(pool_size,pool_size)),
        Conv2D(512, kernel, kernel, activation='relu', border_mode='same'),
        BatchNormalization(),
        Conv2D(512, kernel, kernel, activation='relu', border_mode='same'),
        BatchNormalization(),
        Conv2D(512, kernel, kernel, activation='relu', border_mode='same'),
        BatchNormalization(),

        UpSampling2D(size=(pool_size,pool_size)),
        Conv2D(512, kernel, kernel, activation='relu', border_mode='same'),
        BatchNormalization(),
        Conv2D(512, kernel, kernel, activation='relu', border_mode='same'),
        BatchNormalization(),
        Conv2D(256, kernel, kernel, activation='relu', border_mode='same'),
        BatchNormalization(),

        UpSampling2D(size=(pool_size,pool_size)),
        Conv2D(256, kernel, kernel, activation='relu', border_mode='same'),
        BatchNormalization(),
        Conv2D(256, kernel, kernel, activation='relu', border_mode='same'),
        BatchNormalization(),
        Conv2D(128, kernel, kernel, activation='relu', border_mode='same'),
        BatchNormalization(),

        UpSampling2D(size=(pool_size,pool_size)),
        Conv2D(128, kernel, kernel, activation='relu', border_mode='same'),
        BatchNormalization(),
        Conv2D(64, kernel, kernel, activation='relu', border_mode='same'),
        BatchNormalization(),

        UpSampling2D(size=(pool_size,pool_size)),
        Conv2D(64, kernel, kernel, activation='relu', border_mode='same'),
        BatchNormalization(),
        Conv2D(n_labels, 1, 1, border_mode='valid'),
        BatchNormalization(),
    ]
    return decoding_layers

def getFilter()
    filter = models.Sequential()
    filter.add(Layer(input_shape=(3, img_h, img_w)))

    for l in getEncoder():
        filter.add(l)
    for l in getDecoder():
        filter.add(l)
    filter.add(Reshape((n_labels, img_h*img_w), input_shape=(n_labels, img_h, img_w)))
    filter.add(Permute((2, 1)))
    filter.add(Activation('softmax'))

    return filter