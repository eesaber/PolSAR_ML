# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
np.random.seed(1337) # for reproducibility
from keras import backend as K
from keras import callbacks, optimizers, utils
from scipy.io import loadmat, savemat
from model_4 import create_model


seg_cnn = create_model(img_h, img_w, x_train.shape[-1])
lr = 1 # change to 0.05
decay = 0.0
print(colored('@--------- Parameters ---------@','green'))
print('batch size: '+str(batch_size))
print('learning rate: '+str(lr))
print('decay:'+str(decay))
print('input vector: '+input_vector)
print(colored('@------------------------------@','green'))
if input_vector == '(4)':
    print('a')
    # optimizer = optimizers.Adagrad(lr=lr, epsilon=None, decay=decay)
    optimizer = optimizers.adadelta(lr=lr, rho=0.95, decay=decay)
    # optimizer = optimizers.SGD(lr=lr, momentum=0.9, decay=decay, nesterov=False)
else:
    # optimizer = optimizers.SGD(lr=0.01, momentum=0.9, decay=0.001, nesterov=False)
    optimizer = optimizers.adadelta(lr=lr, rho=0.95, decay=decay)


seg_cnn.compile(optimizer=optimizer, loss='binary_crossentropy',
    metrics=['accuracy'])
tb = callbacks.TensorBoard(
    log_dir=path['log']+'winter/',
    batch_size=batch_size,
    histogram_freq=0,
    write_graph=True,
    write_images=True)
earlystop = callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=1e-4, 
    patience=10)
ckp = callbacks.ModelCheckpoint(
    path['model']+'my_model_'+str(epochs)+'_'+input_vector[1]+'.h5', # file path
    monitor='val_loss',
    verbose=0,
    save_best_only=True, save_weights_only=False, mode='auto', period=1)

#%% training
seg_cnn.summary()