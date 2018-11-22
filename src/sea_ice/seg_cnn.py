# -*- coding: utf-8 -*-
#!/usr/bin/env python3
from myUtility import get_path
from model_3 import create_model
import os

from scipy.io import loadmat, savemat
import numpy as np
from keras import utils, optimizers
from keras import backend as K
from keras import callbacks

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

# read training data
path = get_path()
input_vector = '(4)'
y_train = np.array(loadmat(path['aug']+'y_train_070426_3.mat')['y_train'])
x_train = np.array([])
for filename in sorted(os.listdir(path['aug'])): 
    if input_vector not in filename:
        continue
    print(filename)
    if 'x_train' in filename:
        x = loadmat(path['aug']+filename)
        temp_x = np.array(x['x_train'])
        if x_train.size == 0:
            x_train = temp_x
        else:
            x_train = np.concatenate((x_train, temp_x), axis=0)
        print(x_train.shape)

# read validation data
x_val = np.array(loadmat(path['val']+'x_val_070426_3'+input_vector+'.mat')['x_val'])
y_val = np.array(loadmat(path['val']+'y_val_070426_3.mat')['y_val'])

#%% imput data and setting
n_labels = 2
batch_size = 20
epochs = 10
img_h, img_w = x_train.shape[1], x_train.shape[2]
y_train = utils.to_categorical(y_train, n_labels).astype('float32')
print(y_train.shape)

#%% CNN 
seg_cnn = create_model(img_h, img_w, x_train.shape[-1])
if input_vector == '(3)':
    # optimizer = optimizers.Adagrad(lr=1, epsilon=None, decay=0.001)
    # optimizer = optimizers.adadelta(lr=1.0, rho=0.95, decay=0)
    optimizer = optimizers.SGD(lr=0.1, momentum=0.9, decay=0.001, nesterov=False)
else:
    optimizer = optimizers.SGD(lr=0.01, momentum=0.9, decay=0.001, nesterov=False)

seg_cnn.compile(optimizer=optimizer, loss='binary_crossentropy',
    metrics=['accuracy'])
tb = callbacks.TensorBoard(
    log_dir=path['log'],
    batch_size=100,
    histogram_freq=0,
    write_graph=True,
    write_images=True)
earlystop = callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=1e-4, 
    patience=2)

#%% training
input_vector = input_vector[1:2]
seg_cnn.fit(x_train, y_train,
    batch_size=batch_size,
    validation_data=(x_val, y_val),
    verbose=1,
    epochs=epochs,
    shuffle=True,
    callbacks=[tb])
seg_cnn.save(path['model']+'my_model_'+str(epochs)+'_'+input_vector+'.h5')

print('Session over')