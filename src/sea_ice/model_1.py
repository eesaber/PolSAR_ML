# -*- coding: utf-8 -*-
#!/usr/bin/env python3
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Layer, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation, Reshape, Permute

img_h = 96
img_w = 496
n_labels = 2

def get_encoder():    
    return [
        Conv2D(16, (3, 3), activation='relu', padding='same',
            input_shape=(96,496,3)), 
        BatchNormalization(),
        ZeroPadding2D(padding=(1, 1)),
        Conv2D(16, (3, 3), activation='relu', padding='valid'), #"valid" means "no padding"
        BatchNormalization(),
        MaxPooling2D((2, 2), padding='same'),

        ZeroPadding2D(padding=(1, 1)),
        Conv2D(32, (3, 3), activation='relu', padding='valid'),
        BatchNormalization(),
        ZeroPadding2D(padding=(1, 1)),
        Conv2D(32, (3, 3), activation='relu', padding='valid'),
        BatchNormalization(),
        MaxPooling2D((2, 2), padding='same'),

        ZeroPadding2D(padding=(1, 1)),
        Conv2D(32, (3, 3), activation='relu', padding='valid'),
        BatchNormalization(),
        MaxPooling2D((2, 2), padding='same')
    ]

def get_decoder():
    return [
        UpSampling2D((2, 2)),
        ZeroPadding2D(padding=(1, 1)),
        Conv2D(32, (3, 3), activation='relu', padding='valid'),
        BatchNormalization(),
        
        UpSampling2D((2, 2)),
        ZeroPadding2D(padding=(1, 1)),
        Conv2D(32, (3, 3), activation='relu', padding='valid'),
        BatchNormalization(),
        ZeroPadding2D(padding=(1, 1)),
        Conv2D(32, (3, 3), activation='relu', padding='valid'),
        BatchNormalization(),
        
        UpSampling2D((2, 2)),
        ZeroPadding2D(padding=(1, 1)),
        Conv2D(16, (3, 3), activation='relu', padding='valid'),
        BatchNormalization(),
        ZeroPadding2D(padding=(1, 1)),
        Conv2D(16, (3, 3), activation='relu', padding='valid'),
        BatchNormalization(),
        # connect to label
        Conv2D(n_labels, (1, 1), border_mode='valid'),
        # Reshape((n_labels, img_h*img_w), input_shape=(2,img_h,img_w)),
        Reshape((n_labels, img_h*img_w)),
        Permute((2, 1)),
        Activation('softmax')
    ]

def create_model():
    model = Sequential()
    encoder = get_encoder()
    decoder = get_decoder()
    for l in encoder:
        model.add(l)
    for l in decoder:
        model.add(l)
    return model