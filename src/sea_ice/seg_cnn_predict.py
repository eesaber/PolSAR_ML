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

if 1:
    mat_dict = loadmat(file_path+'image_070426_3_(3).mat')
    x_train = np.array(mat_dict['im'])
    x_train = x_train.reshape((624,4608,3),order='F')
    x_train = resize(x_train, (96, 496, x_train.shape[-1]), anti_aliasing=True)
else:
    mat_dict = loadmat(file_path+'image_070426_3_(5).mat')
    x_train = np.array(mat_dict['map'])
    x_train = x_train.reshape((96,496,3),order='F')

plt.imshow(x_train, aspect='auto',cmap= colors.ListedColormap(np.array([[0,120,0],[180,100,50]])/255))
plt.gca().invert_yaxis()
plt.gca().set_axis_off()
plt.show()

x_train = np.expand_dims(x_train, axis=0)


#%% Load the exist model and predict 
exist_model = load_model(model_path+'my_model_40.h5')
y_hat = exist_model.predict(x_train, verbose=1)

#y_hat = (y_hat.reshape(x_train.shape[0],96,496,2))[:,:,:,1]>0.5
y_hat = y_hat.reshape((96,496,2))[:,:,0]
savemat(file_path+'y_hat_final', {'gt': y_hat}, appendmat=False)

#
plt.imshow(y_hat, 
    aspect='auto')
    #cmap= colors.ListedColormap(np.array([[0,120,0],[180,100,50]])/255))
plt.gca().invert_yaxis()
plt.gca().set_axis_off()
plt.savefig(output_path+'/070426_3_cnn.jpg',
            dpi=300,          
            bbox_inches='tight')

print('Session over')