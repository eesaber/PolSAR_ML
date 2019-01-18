# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
np.random.seed(1337) # for reproducibility
from termcolor import colored
from keras import backend as K
from keras import callbacks, optimizers, utils
from scipy.io import loadmat, savemat
from read import readslc
from get_filter import getFilter

x_train = readslc('/home/akb/下載/Haywrd_23501_18039_014_180801_L090HH_CX_01.slc',
                numel=2*9900*95000,
                offset=2*9900*95000*4)
filter = getFilter(img_h=1024,img_w=1024,dim=9)
batch_size = 20
epochs = 20
lr = 1 # change to 0.05
decay = 0.0
print(colored('@--------- Parameters ---------@','green'))
print('batch size: '+str(batch_size))
print('learning rate: '+str(lr))
print('decay:'+str(decay))
# print('input vector: '+input_vector)
print(colored('@------------------------------@','green'))

# optimizer = optimizers.Adagrad(lr=lr, epsilon=None, decay=decay)
optimizer = optimizers.adadelta(lr=lr, rho=0.95, decay=decay)
# optimizer = optimizers.SGD(lr=lr, momentum=0.9, decay=decay, nesterov=False)

filter.compile(optimizer=optimizer, loss='binary_crossentropy',
    metrics=['accuracy'])
'''
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
'''

#%% training
filter.summary()

filter.fit(x_train_noisy, x_train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(x_test_noisy, x_test))
