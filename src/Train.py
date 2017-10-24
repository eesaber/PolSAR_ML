#!/usr/bin/env python3
# -*- coding: utf-8 -*-  

from FileIO import read

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model,Sequential

import numpy as np
import matplotlib.pyplot as plt

if __name__=='__main__':
    # Loading data and cut the image into pieces 每張小圖的尺寸應該要是一個建築物大小左右。
    s = read()
    scat = s
# Model 
    # Encoding 
    x = Dense(128, activation='relu')(scat)
    x = Dense(64, activation='relu')(x)
    encode = Dense(4, activation='relu')(x)
    # Decoding
    x = Dense(4, activation='relu')(encode)
    decoded = Dense(64, activation='relu')(x)

    autoencoder = Model(scat, decoded) # Autoencoder model
    
    encoder = Model(scat, encode) # Encoder model

    encoded_input = Input(shape=(4,))
    decoder_layer = autoencoder.layers[-1]
    decoder = Model(encoded_input, decoder_layer(encoded_input)) # Decoder model

    #Training part    
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    autoencoder.fit(x_train, x_train, epochs=100, batch_size=256, shuffle=True,
                validation_data=(x_test, x_test))
    model.save('autoencoder.h5') 
    
    '''
    for :
        # display original
        ax = plt.subplot(2, n, i)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
    '''