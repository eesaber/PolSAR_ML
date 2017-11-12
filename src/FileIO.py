#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os.path

def read(argv='single'):
    opt = {'double':'Covariance_d.mat', 'single':'Covariance.mat', 'sparse':'Covariance_ds.mat','test':'test.mat'}
    p = {'/media/akb/2026EF9426EF696C/raw_data/PiSAR2_07507_13170_009_131109_L090_CX_01_grd/', '../data/'}
    for pat in p:
        if os.path.isfile(pat+opt[argv]):
            f =  h5py.File(pat+opt[argv],'r')
            s = {}
            for k,v in f.items():
                s[k] = np.array(v)
            return s
    print ("Training data doesn't exist. The process is closed...")
    sys.exit()

def load_cut():
    return {'train':(2400,2800,7200,8200), 'test':(1)}

if __name__=='__main__':
    # DEBUG 
    s = read('test')
    s['hh_hh'][s['hh_hh'] == 0] = 0.0000001
    plt.figure(1)
    #plt.imshow(10*np.log10(s['hh_hh'][2800:3400, 8300:8855].T), cmap='jet', aspect='auto', clim = (-30,20))
    plt.imshow(10*np.log10(s['hh_hh'].T[2400:2800, 7200:8200]), cmap='jet', aspect='auto', clim = (-30,20))
 
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.title('$S^2_{hh} (dB)$', fontsize=24)
    plt.show()
    