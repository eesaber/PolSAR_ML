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
input_vector = '4'
region = '3'
season = 'winter'
print('input_vector: '+ input_vector)
print('season: '+season)
if season == 'winter':
    # input_vector = '6'
    f_name = path['val']+'x_val_070426_'+region+'_'+input_vector+'.mat'
    print('region: '+ region)
    # input_vector = '1'
    exist_model = load_model('/home/akb/Code/PolSAR_ML/model_select/'+ 'winter_' + input_vector+ '.h5')
    # exist_model = load_model(path['model']+'my_model_100_'+input_vector+'.h5')
    # exist_model = load_model(path['model']+'my_model_n_60_'+input_vector+'.h5')
else:
    f_name = path['val']+'x_val_090811_'+input_vector+'.mat'
    exist_model = load_model(path['model']+'my_model_s_8_'+input_vector+'.h5')
    
x_train = np.array(loadmat(f_name)['x_val'])
print(x_train.shape)
y_hat = exist_model.predict(x_train, verbose=0)

if season == 'winter':
    savemat(path['output']+'y_hat_070426_'+region+'_'+input_vector+'.mat',
        {'y_hat': y_hat}, appendmat=False)
else:
    savemat(path['output']+'y_hat_090811_'+input_vector+'.mat',
        {'y_hat': y_hat}, appendmat=False)

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