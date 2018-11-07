# -*- coding: utf-8 -*-
#!/usr/bin/env python3
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import os.path
from sklearn.neural_network import MLPClassifier


#%% read image
work_path = os.path.dirname(os.path.realpath(__file__))
work_path = work_path[0:work_path.find('src')]
file_path = work_path+'data/final_version/'
model_path =   work_path+'model/'

#f_name_x = 'image_070426_3_(4).mat'
#f_name_y = 'mask_070426_3.mat'
f_name_x = 'image_090811_(2).mat'
f_name_y = 'mask_090811.mat'
#%%
if os.path.isfile(file_path+f_name_x):
    mat_dict = loadmat(file_path+f_name_x)
    x = np.array(mat_dict['im'])
if os.path.isfile(file_path+f_name_y):
    mat_dict = loadmat(file_path+f_name_y)
    y = np.array(mat_dict['gt'])

#%% reshape image and label
y_train = y.reshape((-1, 1), order='F').squeeze()
'''
#x = x.reshape((-1, x.shape[2]))
p = np.random.permutation(y.size)
train_set_size = 2700000
print("Size of traing set: %f %%" % (100*train_set_size/y.size))
x_train = x[p[:train_set_size],:]
y_train = y[p[:train_set_size]]

x_test = x[p[train_set_size:],:]
y_test = y[p[train_set_size:]]
'''
#%% NN 
clf = MLPClassifier(solver='sgd', alpha=1e-5, activation='relu',
                learning_rate_init = 0.1,
                batch_size=100,
                hidden_layer_sizes=(20,30,10), 
                random_state=1,
                validation_fraction=0.05,
                early_stopping=True,
                shuffle=True,
                tol = 1e-8,
                verbose=True)
clf.fit(x, y_train)

y_hat = clf.predict(x)
y_hat = y_hat.reshape((624, -1), order='F')
savemat(file_path+'y_hat_090811.mat',
    {"y_hat" : y_hat},
    appendmat=False)
#%% 
Mm = np.sum((y_hat==1)*(y==1))/y.size
Mf = np.sum((y_hat==1)*(y==0))/y.size
Fm = np.sum((y_hat==0)*(y==1))/y.size
Ff = np.sum((y_hat==0)*(y==0))/y.size

print("Total Accuracy: %f" % (np.sum(y==y_hat)/y_hat.size))
print("Confusion matrix:\n pred/true: MYI  FYI\n MYI     %.3f %.3f\n FYI     %.3f %.3f \n"
 % (Mm, Mf, Fm, Ff))
#%%
plt.figure(1)
plt.imshow(y, aspect='auto',cmap= colors.ListedColormap(np.array([[0,120,0],[180,100,50]])/255))
plt.gca().invert_yaxis()
plt.gca().get_xaxis().set_ticks([])
plt.gca().get_yaxis().set_ticks([])
plt.gca().set_axis_off()
plt.show()
plt.savefig('/home/akb/Code/PolSAR_ML/output/label_070428_NN.jpg',
            dpi=300,
            bbox_inches='tight')
            

