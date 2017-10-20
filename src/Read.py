# -*- coding: utf-8 -*-
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

def Read():
    path = '/media/akb/2026EF9426EF696C/raw_data/PiSAR2_07507_13170_009_131109_L090_CX_01_grd/'
    #path = '../data/'
    with open(path+'Covariance.mat', 'r') as file:
        sig = sio.loadmat(file)
        return sig

if __name__=='__main__':
    s = Read()
    #sig = {'hh_hh': np.random.random((100,100))}
    plt.figure(1)
    plt.imshow(10*np.log10(s['hh_hh']), cmap='jet', extent=[0,100,0,1], aspect='auto')
    plt.gca().invert_yaxis()
    plt.clim((-30,20))
    plt.colorbar()
    plt.title('$S^2_{hh} (dB)$')
    plt.show()