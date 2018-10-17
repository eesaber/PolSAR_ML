# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import scipy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import h5py
import os.path

from sklearn.neural_network import MLPClassifier

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import Dropout, Activation, Flatten
from keras.models import Model, Sequential
from keras import backend as K
from keras.preprocessing import image


#%% tensorflow setting
if 'tensorflow' == K.backend():
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    config.gpu_options.visible_device_list = "0"
    set_session(tf.Session(config=config))

#%% read image
#path = '/home/akb/Code/PolSAR_ML/data/'
path = 'C:/Code/PolSAR_ML/data/'

with h5py.File(path+'x.mat','r') as f:
    f.keys()
    #for k, v in f.items():
    #    x_train = np.array(v)
'''
if os.path.isfile(path+'y.mat'):
    mat_dict = scipy.io.loadmat(path+'y.mat')
    y_train = np.array(mat_dict['y'])
'''
#%% imput data and setting
data_augmentation = 0
batch_size = 32
epochs = 100
y_train = y_train.astype('float32')


#%% NN Architechture

seg_cnn = Sequential()
seg_cnn.add(Conv2D(16, (3, 3), activation='relu', padding='same',
                 input_shape=x_train.shape[1:]))
seg_cnn.add(MaxPooling2D((2, 2), padding='same'))
seg_cnn.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
seg_cnn.add(MaxPooling2D((2, 2), padding='same'))
seg_cnn.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
seg_cnn.add(MaxPooling2D((2, 2), padding='same'))

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

seg_cnn.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
seg_cnn.add(UpSampling2D((2, 2)))
seg_cnn.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
seg_cnn.add(UpSampling2D((2, 2)))
seg_cnn.add(Conv2D(16, (3, 3), activation='relu'))
seg_cnn.add(UpSampling2D((2, 2)))
seg_cnn.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))
seg_cnn.add(Activation('softmax'))

seg_cnn.compile(optimizer='adadelta', loss='binary_crossentropy') 


#%% training 
if not data_augmentation:
    seg_cnn.fit(x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True)
else:
    datagen = image.ImageDataGenerator(samplewise_center=True,
    rotation_range=30,
    horizontal_flip=True,
    vertical_flip=True)