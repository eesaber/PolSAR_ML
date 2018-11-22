# -*- coding: utf-8 -*-
#!/usr/bin/env python3
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Layer, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation, Reshape, Permute

n_labels = 2
kernel = (3,3)
zero_padding = (1,1)
def get_encoder(img_h, img_w, dim):
    return [
        Conv2D(32, kernel, activation='relu', padding='same',
            input_shape=(img_h,img_w,dim)),
        BatchNormalization(),
        ZeroPadding2D(padding=zero_padding),
        Conv2D(32, kernel, activation='relu', padding='valid'), #"valid" means "no padding"
        BatchNormalization(),
        MaxPooling2D((2, 2), padding='same'),

        ZeroPadding2D(padding=zero_padding),
        Conv2D(64, kernel, activation='relu', padding='valid'),
        BatchNormalization(),
        ZeroPadding2D(padding=zero_padding),
        Conv2D(64, kernel, activation='relu', padding='valid'),
        BatchNormalization(),
        MaxPooling2D((2, 2), padding='same'),
    ]

def get_decoder():
    return [
        UpSampling2D((2, 2)),
        ZeroPadding2D(padding=zero_padding),
        Conv2D(64, kernel, activation='relu', padding='valid'),
        BatchNormalization(),
        ZeroPadding2D(padding=zero_padding),
        Conv2D(64, kernel, activation='relu', padding='valid'),
        BatchNormalization(),

        UpSampling2D((2, 2)),
        ZeroPadding2D(padding=zero_padding),
        Conv2D(32, kernel, activation='relu', padding='valid'),
        BatchNormalization(),
        ZeroPadding2D(padding=zero_padding),
        Conv2D(32, kernel, activation='relu', padding='valid'),
        BatchNormalization(),
        
        # connect to label
        #Conv2D(2, (1, 1), activation='sigmoid', padding='valid'),
        Conv2D(2, (1, 1), activation='softmax', padding='valid'),
    ]

def create_model(img_h, img_w, dim):
    model = Sequential()
    encoder = get_encoder(img_h, img_w, dim)
    decoder = get_decoder()
    for l in encoder:
        model.add(l)
    for l in decoder:
        model.add(l)
    return model