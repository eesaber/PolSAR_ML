# -*- coding: utf-8 -*-
#!/usr/bin/env python3
from scipy.io import loadmat, savemat
import numpy as np
import os
from keras.models import load_model

#%% read file
work_path = os.path.dirname(os.path.realpath(__file__))
work_path = work_path[0:work_path.find('src')]
file_path = work_path+'data/'
model_path = work_path+'model/'

if os.path.isfile(file_path+'im_070426_3_cnn_final.mat'):
    mat_dict = loadmat(file_path+'im_070426_3_cnn_final.mat')
    x_train = np.array(mat_dict['generate_cnn_label'])    

#%% Load the exist model and predict 
exist_model = load_model(model_path+'my_model_100.h5')
y_hat = exist_model.predict(x_train, verbose=1)

gt = {}
gt['gt'] = (y_hat.reshape(x_train.shape[0],96,496,2))[:,:,:,1]>0.5
savemat(file_path+'y_hat_final', gt, appendmat=False)

print('Session over')