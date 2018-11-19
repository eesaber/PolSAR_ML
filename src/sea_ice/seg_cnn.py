# -*- coding: utf-8 -*-
#!/usr/bin/env python3
from model_3 import create_model

from scipy.io import loadmat, savemat
from skimage.transform import resize
import numpy as np
import os
from myImageGenerator import myImageGenerator
import matplotlib.pyplot as plt
from matplotlib import colors

from keras import utils, optimizers
from keras import backend as K
from keras import callbacks
from keras.models import load_model
#%% tensorflow setting
eat_all = 0
if not eat_all and 'tensorflow' == K.backend():
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.35
    config.gpu_options.visible_device_list = "0"
    set_session(tf.Session(config=config))

#%% read file
work_path = os.path.dirname(os.path.realpath(__file__))
work_path = work_path[0:work_path.find('src')]
#%%
file_path = work_path+'data/'
log_path = work_path+'log/'
if not os.path.exists(file_path):
    os.makedirs(file_path)
#%% read file

model_path = work_path+'model/'
if not os.path.exists(model_path):
    os.makedirs(model_path)

augmentation_file_path = work_path+'data_aug/'
if not os.path.exists(augmentation_file_path):
    os.makedirs(augmentation_file_path)


#%%
input_opt = 0
if input_opt == 0:
    if os.path.isfile(augmentation_file_path+'x_train_070428_3.mat'):
        mat_dict = loadmat(augmentation_file_path+'x_train_070428_3.mat')
        x_train = np.array(mat_dict['x_train'])
        mat_dict = loadmat(augmentation_file_path+'y_train_070428_3.mat')
        y_train = np.array(mat_dict['y_train'])
else:
    mat_dict = loadmat(file_path+'mask_070426_3_(2).mat')
    y_train = np.array(mat_dict['gt']).reshape((-1, 496),order='F')
    if input_opt == 1:
        mat_dict = loadmat(file_path+'image_070426_3_(3).mat')
        x_train = np.array(mat_dict['im'])
        x_train = x_train.reshape((624,4608,x_train.shape[1]),order='F')
        x_train = resize(x_train, (96, 496), anti_aliasing=True)
        x_train = np.expand_dims(x_train, axis=0)
    else:
        mat_dict = loadmat(file_path+'image_070426_3_(5).mat')
        x_train = np.array(mat_dict['qq'])
        x_train = np.expand_dims(x_train, axis=0)
        
print(x_train.shape)
#print(y_train.shape)
x_train = 2*x_train

#%% imput data and setting
n_labels = 2
batch_size = 5
epochs = 10
img_h, img_w = 496, 496
y_train = utils.to_categorical(y_train, n_labels).astype('float32')
print(y_train.shape)
'''
plt.imshow(y_train[0,:,:,0], aspect='auto',cmap= colors.ListedColormap(np.array([[0,120,0],[180,100,50]])/255))
plt.gca().invert_yaxis()
plt.gca().set_axis_off()
plt.show()
'''
#%% CNN 
seg_cnn = create_model()
optimizer = optimizers.SGD(lr=1, momentum=0.9, decay=0.001, nesterov=False)
#optimizer = optimizers.adadelta(lr=0.01, decay=0)
#optimizer = optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.001)
seg_cnn.compile(optimizer=optimizer, loss='binary_crossentropy',
    metrics=['accuracy'])
tb = callbacks.TensorBoard(
    log_dir=log_path,
    batch_size=50,
    histogram_freq=0,
    write_graph=True,
    write_images=True)
earlystop = callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=1e-4, 
    patience=2)

#%% training
#do_train = int(input('Train? [1/0]:'))
do_train = 1
if do_train:    
    # seg_cnn.fit(x_train, y_train.reshape((x_train.shape[0],img_h*img_w,n_labels)),
    seg_cnn.fit(x_train, y_train,
        batch_size=batch_size,
        verbose=1,
        epochs=epochs,
        shuffle=True,
        callbacks=[tb])
    seg_cnn.save(model_path+'my_model_'+str(epochs)+'.h5')
else:
    seg_cnn = load_model(model_path+'my_model_'+str(epochs)+'.h5')

'''
#%%
y_train_hat = seg_cnn.predict(x_train)
plt.imshow(y_train_hat[0,:,:,0], aspect='auto')
# plt.colorbar()
plt.gca().invert_yaxis()
plt.gca().set_axis_off()
plt.show()
plt.imshow(y_train_hat[0,:,:,1], aspect='auto')
# plt.colorbar()
plt.gca().invert_yaxis()
plt.gca().set_axis_off()
plt.show()
'''
print('Session over')