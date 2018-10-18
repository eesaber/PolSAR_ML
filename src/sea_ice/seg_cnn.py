# -*- coding: utf-8 -*-
#!/usr/bin/env python3
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import os

from keras import utils
from keras import backend as K
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation, Reshape, Permute
from keras.preprocessing import image


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
model_path =   work_path+'model/'

if os.path.isfile(file_path+'x_1.mat'):
    mat_dict = loadmat(file_path+'x_1.mat')
    x_train_1 = np.array(mat_dict['x_1'])    
if os.path.isfile(file_path+'x_2.mat'):
    mat_dict = loadmat(file_path+'x_2.mat')
    x_train_2 = np.array(mat_dict['x_2'])
x_train = np.concatenate((x_train_1, x_train_2), axis=0)

if os.path.isfile(file_path+'y.mat'):
    mat_dict = loadmat(file_path+'y.mat')
    y_train = np.array(mat_dict['y'])

#%%
'''
plt.figure()
plt.imshow(x_train[1,:,:,:])
plt.gca().invert_yaxis()    
plt.show()
'''
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


#%% training
do_train = int(input('Train? [1/0]:'))
if do_train:
    data_augmentation = int(input('Data augmentation? [1/0]:'))
    if not data_augmentation:
        seg_cnn.fit(x_train, y_train.reshape((7260,47616,2)),
            batch_size=batch_size,
            epochs=epochs,
            shuffle=True)
        seg_cnn.save(model_path+'my_model_'+str(epochs)+'.h5')
    else:
        datagen = image.ImageDataGenerator(samplewise_center=False,
        rotation_range=30,
        horizontal_flip=True,
        vertical_flip=True)
        datagen.fit(x_train)
        seg_cnn.fit_generator(datagen.flow(x_train, y_train.reshape((7260,47616,2)),
                                        batch_size=batch_size),
        epochs=epochs)
        seg_cnn.save(model_path+'my_model_'+str(epochs)+'_aug.h5')

    y_hat = seg_cnn.predict(x_train[0:3630,:,:,:], verbose=1)
else:
    exist_model = load_model(model_path+'my_model_100.h5')
    y_hat = exist_model.predict(x_train[0:3630,:,:,:], verbose=1)
    
gt = y_hat.reshape(3630,96,496,2)
gt = (gt[:,:,:,1]>0.5)
score = np.sum(np.sum(np.sum(np.equal(y_train[0:3630],gt))))/gt.size
print('Train accuracy: ', score)
#%%
'''
img_num = 1
plt.figure()
plt.subplot(2,1,1)
plt.imshow(y_train[img_num,:,:])
plt.subplot(2,1,2)
plt.imshow(gt[img_num,:,:])
plt.show()
'''
print('Session over')