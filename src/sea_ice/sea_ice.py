# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import scipy

import h5py
import os.path

def readImage(argv):
    path = 'C:\Code\PolSAR_ML\data\image.mat'
    if os.path.isfile(path):
        mat_dict = scipy.io.loadmat(path)
        s = np.array(mat_dict['im'])            
        return s
    else:
        print('ERROR: File does not exist.')


#%% read image
raw = readImage('image.mat')


#label = read('label.mat')
print(type(raw))
print(raw.shape)

#%%
plt.figure(1)
plt.imshow(raw)
#plt.gca().invert_yaxis()    
plt.show()

#%% reshape image and label
x = raw.reshape((-1, s.shape[2]), order='F')
#y = label.reshape((-1, 1), order='F')
# recale 
x_min = np.amin(x, axis=0)
x_max = np.amax(x, axis=0)
u = 0
l = 1
x = (x-x_min)/(x_max-xmin)*(u-l) + l
'''
#%% NN 
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, early_stopping=True, activation='relu',
                hidden_layer_sizes=(5, 2), random_state=1, 
                verbose=True, )
clf.fit(x, y)
'''