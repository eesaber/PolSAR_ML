# -*- coding: utf-8 -*-
#!/usr/bin/env python3
from scipy.io import loadmat, savemat
import numpy as np
import os
from keras.models import load_model
from skimage.transform import resize
import matplotlib.pyplot as plt
from matplotlib import colors

#%% read file
work_path = os.path.dirname(os.path.realpath(__file__))
work_path = work_path[0:work_path.find('src')]
file_path = work_path+'data/'
output_path = work_path+'output/'
model_path = work_path+'model/'
#%%
if 0:
    mat_dict = loadmat(file_path+'image_070426_3_(3).mat')
    x_train = np.array(mat_dict['im'])
    x_train = x_train.reshape((624,4608,3),order='F')
    x_train = resize(x_train, (96, 496, x_train.shape[-1]), anti_aliasing=True)
    mat_dict = loadmat(file_path+'mask_070426_3.mat')
    y_train = np.array(mat_dict['gt'])
    y_train = y_train.reshape((624,4608),order='F')
    #y_train = resize(y_train, (96, 496), anti_aliasing=True) > 0.5
else:
    mat_dict = loadmat(file_path+'image_070426_3_(5).mat')
    x_train = np.array(mat_dict['qq'])
    x_train = np.expand_dims(x_train, axis=0)

x_train = x_train*2
'''
plt.imshow(x_train[0,:,:,0], aspect='auto',cmap= colors.ListedColormap(np.array([[0,120,0],[180,100,50]])/255))
plt.gca().invert_yaxis()
plt.gca().set_axis_off()
plt.show()
'''
print(x_train.shape)

#%% Load the exist model and predict 
exist_model = load_model(model_path+'my_model_5.h5')
y_hat = exist_model.predict(x_train, verbose=0)

#y_hat = (y_hat.reshape(x_train.shape[0],96,496,2))[:,:,:,1]>0.5
y_hat = y_hat.reshape((496,496,2))
#y_hat = y_hat[:,:,1]>y_hat[:,:,0]
savemat(file_path+'y_hat_final', {'gt': y_hat}, appendmat=False)

#
plt.imshow(y_hat[:,:,1], 
    aspect='auto')
    #cmap= colors.ListedColormap(np.array([[0,120,0],[180,100,50]])/255))
plt.gca().invert_yaxis()
#plt.gca().set_axis_off()
plt.savefig(output_path+'/070426_3_cnn.jpg',
            dpi=300,          
            bbox_inches='tight')
plt.show()
print('Session over')