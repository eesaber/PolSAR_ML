# -*- coding: utf-8 -*-
#!/usr/bin/env python3
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation, Reshape, Permute

def basic_model():
    img_h = 96
    img_w = 496
    n_labels = 2
    seg_cnn = Sequential()
    # encoder
    seg_cnn.add(Conv2D(16, (3, 3), activation='relu', border_mode='same',
                    input_shape=(96,496,3)))
    seg_cnn.add(BatchNormalization())
    seg_cnn.add(MaxPooling2D((2, 2), padding='same'))

    seg_cnn.add(Conv2D(16, (3, 3), activation='relu', border_mode='same'))
    seg_cnn.add(BatchNormalization())
    seg_cnn.add(MaxPooling2D((2, 2), padding='same'))

    seg_cnn.add(Conv2D(32, (3, 3), activation='relu', border_mode='same'))
    seg_cnn.add(BatchNormalization())
    seg_cnn.add(MaxPooling2D((2, 2), padding='same'))

    # decoder
    seg_cnn.add(UpSampling2D((2, 2)))
    seg_cnn.add(Conv2D(32, (3, 3), activation='relu', border_mode='same'))
    seg_cnn.add(BatchNormalization())
    seg_cnn.add(UpSampling2D((2, 2)))
    seg_cnn.add(Conv2D(16, (3, 3), activation='relu', border_mode='same'))
    seg_cnn.add(BatchNormalization())
    # connect to label
    seg_cnn.add(UpSampling2D((2, 2)))
    seg_cnn.add(Conv2D(16, (3, 3), activation='relu',border_mode='same'))
    seg_cnn.add(BatchNormalization())
    seg_cnn.add(Conv2D(n_labels, 1, 1, border_mode='valid'))
    seg_cnn.add(BatchNormalization())

    seg_cnn.add(Reshape((n_labels, img_h*img_w)))
    seg_cnn.add(Permute((2, 1)))
    seg_cnn.add(Activation('softmax'))
    seg_cnn.compile(optimizer='adadelta', loss='categorical_crossentropy')

    return seg_cnn