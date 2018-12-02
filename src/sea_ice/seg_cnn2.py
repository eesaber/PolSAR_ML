# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import os

import numpy as np
np.random.seed(1337) # for reproducibility
from keras import backend as K
from keras import callbacks, optimizers, utils
from scipy.io import loadmat, savemat

from model_4 import create_model
from myUtility import get_path

#%% tensorflow setting
eat_all = 0
if not eat_all and 'tensorflow' == K.backend():
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.7
    config.gpu_options.visible_device_list = "0"
    set_session(tf.Session(config=config))

## read data
path = get_path()
input_vector = '(2)'
# read validation data
x_val = np.array(loadmat(path['val']+'x_val_090811_'+input_vector[1]+'.mat')['x_val'])
y_val = np.array(loadmat(path['val']+'y_val_090811.mat')['y_val'])
# read training data
print('Does not use data aegmentation')
x_train = x_val
y_train = y_val

#%% imput data and setting
n_labels = 2
batch_size = 3
epochs = 8
img_h, img_w = x_train.shape[1], x_train.shape[2]
y_train = utils.to_categorical(y_train, n_labels).astype('float32')
y_val = utils.to_categorical(y_val, n_labels).astype('float32')
print(y_train.shape)

#%% CNN 
lr = 1
decay = 0.00
print('learning rate: '+str(lr))
print('decay:'+str(decay))
print('input vector: '+input_vector)
seg_cnn = create_model(img_h, img_w, x_train.shape[-1])
if input_vector == '(4)':
    # 4:(1, 0.00)
    optimizer = optimizers.adadelta(lr=lr, rho=0.95, decay=decay)
else:
    # optimizer = optimizers.SGD(lr=0.001, momentum=0.9, decay=0.001, nesterov=False)
    optimizer = optimizers.adadelta(lr=lr, rho=0.95, decay=decay)

seg_cnn.compile(optimizer=optimizer, loss='binary_crossentropy',
    metrics=['accuracy'])
tb = callbacks.TensorBoard(
    log_dir=path['log']+'summer/',
    batch_size=batch_size,
    histogram_freq=0,
    write_graph=True,
    write_images=True)
earlystop = callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=1e-4, 
    patience=2)

#%% training
input_vector = input_vector[1]
seg_cnn.fit(x_train, y_train,
    batch_size=batch_size,
    validation_data=(x_val, y_val),
    verbose=1,
    epochs=epochs,
    shuffle=True,
    callbacks=[tb])
seg_cnn.save(path['model']+'my_model_s_'+str(epochs)+'_'+input_vector+'.h5')

print('Session over')
