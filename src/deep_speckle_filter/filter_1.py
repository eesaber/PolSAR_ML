import keras.models as models
from keras.layers.core import Layer, Dropout, Reshape, Permute
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization

from keras import backend as K
import numpy as np


kernel = 3
pad = 1
pool_size = 2
def getEncoder():
    encoding_layers = [
        Conv2D(64, (kernel, kernel), activation='relu', border_mode='same'),
        BatchNormalization(),
        Conv2D(64, (kernel, kernel), activation='relu', border_mode='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        Conv2D(128, (kernel, kernel), activation='relu', border_mode='same'),
        BatchNormalization(),
        Conv2D(128, (kernel, kernel), activation='relu', border_mode='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        Conv2D(256, (kernel, kernel), activation='relu', border_mode='same'),
        BatchNormalization(),
        Conv2D(256, (kernel, kernel), activation='relu', border_mode='same'),
        BatchNormalization(),
        Conv2D(256, (kernel, kernel), activation='relu', border_mode='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        Conv2D(512, (kernel, kernel), activation='relu', border_mode='same'),
        BatchNormalization(),
        Conv2D(512, (kernel, kernel), activation='relu', border_mode='same'),
        BatchNormalization(),
        Conv2D(512, (kernel, kernel), activation='relu', border_mode='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        Conv2D(512, (kernel, kernel), activation='relu', border_mode='same'),
        BatchNormalization(),
        Conv2D(512, (kernel, kernel), activation='relu', border_mode='same'),
        BatchNormalization(),
        Conv2D(512, (kernel, kernel), activation='relu', border_mode='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(pool_size, pool_size)),
    ]
    return encoding_layers

def getDecoder():
    decoding_layers = [
        UpSampling2D(size=(pool_size,pool_size)),
        Conv2D(512, (kernel, kernel), activation='relu', border_mode='same'),
        BatchNormalization(),
        Conv2D(512, (kernel, kernel), activation='relu', border_mode='same'),
        BatchNormalization(),
        Conv2D(512, (kernel, kernel), activation='relu', border_mode='same'),
        BatchNormalization(),

        UpSampling2D(size=(pool_size,pool_size)),
        Conv2D(512, (kernel, kernel), activation='relu', border_mode='same'),
        BatchNormalization(),
        Conv2D(512, (kernel, kernel), activation='relu', border_mode='same'),
        BatchNormalization(),
        Conv2D(256, (kernel, kernel), activation='relu', border_mode='same'),
        BatchNormalization(),

        UpSampling2D(size=(pool_size,pool_size)),
        Conv2D(256, (kernel, kernel), activation='relu', border_mode='same'),
        BatchNormalization(),
        Conv2D(256, (kernel, kernel), activation='relu', border_mode='same'),
        BatchNormalization(),
        Conv2D(128, (kernel, kernel), activation='relu', border_mode='same'),
        BatchNormalization(),

        UpSampling2D(size=(pool_size,pool_size)),
        Conv2D(128, (kernel, kernel), activation='relu', border_mode='same'),
        BatchNormalization(),
        Conv2D(64, (kernel, kernel), activation='relu', border_mode='same'),
        BatchNormalization(),

        UpSampling2D(size=(pool_size,pool_size)),
        Conv2D(64, (kernel, kernel), activation='relu', border_mode='same'),
        BatchNormalization(),
        Conv2D(1, 1, 1, activation='sigmoid', border_mode='valid'),
        BatchNormalization(),
    ]
    return decoding_layers