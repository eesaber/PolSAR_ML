# -*- coding: utf-8 -*-
#!/usr/bin/env python3
from scipy.io import loadmat, savemat
import numpy as np
import os
from myUtility import get_path
from keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib import colors

path = get_path()
#%%
input_vetor = '3'
region = '3'
f_name = path['val']+'x_val_070426_'+region+'_'+input_vetor+'.mat'
x_train = np.array(loadmat(f_name)['x_val'])
print(x_train.shape)

'''
plt.imshow(x_train[0,:,:,0], aspect='auto',cmap= colors.ListedColormap(np.array([[0,120,0],[180,100,50]])/255))
plt.gca().invert_yaxis()
plt.gca().set_axis_off()
plt.show()
'''

#%% Load the exist model and predict 
exist_model = load_model(path['model']+'my_model_10_'+input_vetor+'.h5')
y_hat = exist_model.predict(x_train, verbose=0)

savemat(path['output']+'y_hat_070426_'+region+'_'+input_vetor+'.mat',
    {'y_hat': y_hat}, appendmat=False)

#
'''
plt.imshow(y_hat[:,:,1], 
    aspect='auto',
    cmap= colors.ListedColormap(np.array([[0,120,0],[180,100,50]])/255))
plt.gca().invert_yaxis()
plt.gca().set_axis_off()
plt.savefig(path['output']+'/070426_3_cnn.jpg',
            dpi=300,          
            bbox_inches='tight')
plt.show()
'''
print('Session over')