# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

import sys
import h5py
import os.path

def read(argv):
    path = 'D:\Code\PolSAR_ML\data\image.mat'
    if os.path.isfile(path):
        f =  h5py.File(path,'r')
        s = {}
        for k,v in f.items():
            s[k] = np.array(v)
        return s

if __name__=='__main__':
    #%% read image
    raw = read('image.mat')
    #label = read('label.mat')
    print(type(raw))
    print(raw.shape)

    #%%
    plt.figure(1)
    plt.imshow(raw[:,:,0:2])
    plt.gca().invert_yaxis()
    plt.title('$S^2_{hh} (dB)$', fontsize=24)
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