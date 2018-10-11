# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
from scipy.io import loadmat
from sklearn.manifold import Isomap
import matplotlib.pyplot as plt

path = ['/mnt/d/Code/PolSAR_ML/data/test.mat', 'C:/Code/PolSAR_ML/data/test.mat',
		'/home/akb/Code/PolSAR_ML/data/test.mat']
s = loadmat(path[2])
hh_hh = np.array(s['hh_hh'])
hv_hv = np.array(s['hv_hv'])
vv_vv = np.array(s['vv_vv'])
hh_hv = np.array(s['hh_hv'])
hh_vv = np.array(s['hh_vv'])
hv_vv = np.array(s['hv_vv'])
del s
size_N = hh_hh.size

plt.figure(1)
plt.imshow(10*np.log10(hh_hh), cmap='jet', aspect='auto', clim = (-30,20)) 
plt.gca().invert_yaxis()
plt.colorbar()
plt.title('$S^2_{hh} (dB)$', fontsize=24)
plt.show()
#%%
T_11 = (hh_hh+vv_vv+hh_vv+np.conj(hh_vv))/2.0
T_22 = (hh_hh+vv_vv-hh_vv-np.conj(hh_vv))/2.0
T_33 = 2*hv_hv
T_12 = (hh_hh-vv_vv-hh_vv+np.conj(hh_vv))/2.0
T_13 = hh_hv + np.conj(hv_vv)
T_23 = hh_hv - np.conj(hv_vv)
X_train = np.concatenate((np.reshape(T_11, [1, size_N]), np.reshape(T_22, [1, size_N]), np.reshape(T_33, [1, size_N]),
		np.reshape(T_12.real, [1, size_N])/np.sqrt(2), np.reshape(T_12.imag, [1, size_N])/np.sqrt(2),
		np.reshape(T_13.real, [1, size_N])/np.sqrt(2), np.reshape(T_13.imag, [1, size_N])/np.sqrt(2),
		np.reshape(T_23.real, [1, size_N])/np.sqrt(2), np.reshape(T_23.imag, [1, size_N])/np.sqrt(2)), axis=0).T.real
del T_11, T_22, T_33, T_12, T_13, T_23

model = Isomap(n_components=2, n_jobs=-1)
Y_train = model.fit_transform(X_train)

plt.figure()
plt.scatter(Y_train[:, 0], Y_train[:, 1], cmap=plt.cm.Spectral)
plt.title('Isomap', fontsize=20)
plt.show()