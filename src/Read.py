#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import hdf5storage
import numpy as np
import matplotlib.pyplot as plt
import os.path

def Read(argv='single'):
    opt = {'double':'Covariance_d.mat', 'single':'Covariance.mat', 'sparse':'Covariance_ds.mat'}
    p = {'/media/akb/2026EF9426EF696C/raw_data/PiSAR2_07507_13170_009_131109_L090_CX_01_grd/', '../data/'}
    for pat in p:
        if os.path.isfile(pat+opt[argv]):
            f = hdf5storage.loadmat(pat+opt[argv],'r')
            #print(pat+opt[argv])
            s = {}
            for k,v in f.items():
                print k
                s[k] = np.array(v)
            return s
    print "Trainging data doesn't exist."

if __name__=='__main__':
    s = Read('single')
    #s = {'hh_hh': np.random.random((100,100))}
    plt.figure(1)
    plt.imshow(10*np.log10(s['hh_hh']), cmap='jet', extent=[0,100,0,1], aspect='auto')
    plt.gca().invert_yaxis()
    plt.clim((-30,20))
    plt.colorbar()
    plt.title('$S^2_{hh} (dB)$')
    plt.show()