# -*- coding: utf-8 -*-
#!/usr/bin/env python3
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import numpy as np
import os
from myImageGenerator import myImageGenerator

from keras import utils
from keras import backend as K
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation, Reshape, Permute
from keras.preprocessing import image
from keras.optimizers import SGD, adadelta

#%% tensorflow setting
eat_all = 1
if not eat_all and 'tensorflow' == K.backend():
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    config.gpu_options.visible_device_list = "0"
    set_session(tf.Session(config=config))

#%% read file
work_path = os.path.dirname(os.path.realpath(__file__))
work_path = work_path[0:work_path.find('src')]
file_path = work_path+'data/'
model_path = work_path+'model/'
augmentation_file_path = work_path+'data_aug/'

#%%
x_train, y_train = myImageGenerator()

#%% imput data and setting
batch_size = 16
epochs = 100
n_labels = 2
img_h = 96
img_w = 496
img_h, img_w = y_train.shape[1:]
y_train = y_train.astype('float32')
y_train = utils.to_categorical(y_train, n_labels).astype('float32')

#%% NN Architechture

seg_cnn = Sequential()
# encoder
seg_cnn.add(Conv2D(16, (3, 3), activation='relu', border_mode='same',
                 #input_shape=x_train.shape[1:]))
                 input_shape=(96,496,3)))
seg_cnn.add(BatchNormalization())
seg_cnn.add(Conv2D(16, (3, 3), activation='relu', border_mode='same'))
seg_cnn.add(BatchNormalization())
seg_cnn.add(MaxPooling2D((2, 2), padding='same'))

seg_cnn.add(Conv2D(32, (3, 3), activation='relu', border_mode='same'))
seg_cnn.add(BatchNormalization())
seg_cnn.add(Conv2D(32, (3, 3), activation='relu', border_mode='same'))
seg_cnn.add(BatchNormalization())
seg_cnn.add(MaxPooling2D((2, 2), padding='same'))

seg_cnn.add(Conv2D(32, (3, 3), activation='relu', border_mode='same'))
seg_cnn.add(BatchNormalization())
seg_cnn.add(MaxPooling2D((2, 2), padding='same'))

# decoder
seg_cnn.add(UpSampling2D((2, 2)))
seg_cnn.add(Conv2D(32, (3, 3), activation='relu', border_mode='same'))
seg_cnn.add(BatchNormalization())
seg_cnn.add(Conv2D(32, (3, 3), activation='relu', border_mode='same'))
seg_cnn.add(BatchNormalization())

seg_cnn.add(UpSampling2D((2, 2)))
seg_cnn.add(Conv2D(16, (3, 3), activation='relu', border_mode='same'))
seg_cnn.add(BatchNormalization())
seg_cnn.add(Conv2D(16, (3, 3), activation='relu', border_mode='same'))
seg_cnn.add(BatchNormalization())
# connect to label
seg_cnn.add(UpSampling2D((2, 2)))
seg_cnn.add(Conv2D(16, (3, 3), activation='relu',border_mode='same'))
seg_cnn.add(BatchNormalization())

seg_cnn.add(Conv2D(n_labels, 1, 1, border_mode='valid'))
seg_cnn.add(Reshape((n_labels, img_h*img_w), input_shape=(2,img_h,img_w)))
seg_cnn.add(Permute((2, 1)))
seg_cnn.add(Activation('softmax'))

optimizer = SGD(lr=0.01, momentum=0.9, decay=0.001, nesterov=False)
#optimizer = adadelta(lr=0.01,)
seg_cnn.compile(optimizer=optimizer, loss='categorical_crossentropy',
    metrics=['accuracy']) 


#%% training
do_train = int(input('Train? [1/0]:'))
if do_train:    
    seg_cnn.fit(x_train, y_train.reshape((7260,47616,2)),
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True)
    seg_cnn.save(model_path+'my_model_'+str(epochs)+'.h5')
    
print('Session over')