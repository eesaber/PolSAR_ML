#!/usr/bin/env python3
# -*- coding: utf-8 -*-  

from FileIO import read, get_cut

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model,Sequential
from keras import regularizers

import numpy as np
import matplotlib.pyplot as plt

def direct(code_dim = 4):
    # If 
    # Loading data and cut the image into pieces 每張小圖的尺寸應該要是一個建築物大小左右。
    s = read()
    cut = load_cut()
    x_train = 
    x_test = 
# Model 
    encoding_dim = code_dim
    # Encoding 
    scat = Input(shape=(10*10,))
    x = Dense(128, activation='relu', activity_regularizer=regularizers.l1(10e-5))(scat)
    x = Dense(64, activation='relu', activity_regularizer=regularizers.l1(10e-5))(x)
    encode = Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.l1(10e-5))(x)
    # Decoding
    x = Dense(64, activation='relu')(encode)
    decoded = Dense(128, activation='relu')(x)

    autoencoder = Model(scat, decoded) # Autoencoder model
    encoder = Model(scat, encode) # Encoder model
    encoded_input = Input(shape=(encoding_dim,)) # Encoded holding-place
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
def conv(code_dim = 4):
    1

if __name__=='__main__':
    direct()