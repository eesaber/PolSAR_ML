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
file_path = work_path+'data/'
model_path = work_path+'model/'
output_path = work_path+'output/'

date = '070426'
region = '3'
input_vector = '(1)'
f_train_x = 'image_'+date+'_'+region+'_'+input_vector+'.mat'
f_train_y = 'mask_'+date+'_'+region+'.mat'
#%%
if os.path.isfile(file_path+f_train_x):
    mat_dict = loadmat(file_path+f_train_x)
    x_train = np.array(mat_dict['im'])
if os.path.isfile(file_path+f_train_y):
    mat_dict = loadmat(file_path+f_train_y)
    y = np.array(mat_dict['gt'])

#%% reshape image and label
y_train = y.reshape((-1, 1), order='F').squeeze()
y = y.reshape((624,-1), order='F')
#%% NN 
clf = MLPClassifier(solver='sgd', alpha=1e-3, activation='relu',
                learning_rate_init = 0.1,
                batch_size=100,
                hidden_layer_sizes=(20,30,10), 
                random_state=1,
                validation_fraction=0.1,
                early_stopping=True,
                shuffle=True,
                tol = 1e-8,
                verbose=True)
clf.fit(x_train, y_train)

y_train_hat = clf.predict(x_train)
y_train_hat = y_train_hat.reshape((624, -1), order='F')
savemat(output_path+'y_hat_'+date+'_'+region+'_nn_'+input_vector[1]+'.mat',
    {"y_hat" : y_train_hat},
    appendmat=False)
#%% 
Mm = np.sum((y_train_hat==1)*(y==1))/y.size
Mf = np.sum((y_train_hat==1)*(y==0))/y.size
Fm = np.sum((y_train_hat==0)*(y==1))/y.size
Ff = np.sum((y_train_hat==0)*(y==0))/y.size

print("Total Accuracy: %f" % (np.sum(y==y_train_hat)/y_train_hat.size))
print("Confusion matrix:\n pred/true: MYI  FYI\n MYI     %.3f %.3f\n FYI     %.3f %.3f \n"
 % (Mm, Mf, Fm, Ff))
#%%
'''
plt.figure(1)
plt.imshow(y_train_hat, aspect='auto',cmap= colors.ListedColormap(np.array([[0,120,0],[180,100,50]])/255))
plt.gca().invert_yaxis()
plt.gca().set_axis_off()
plt.savefig(output_path+'label_'+date+'_'+region+'_nn.jpg',
            dpi=300,
            bbox_inches='tight')
plt.show()
'''

#%% Testing 
testing = 1
if testing:
    region = '2'
    f_test_x = 'image_'+date+'_'+region+'_'+input_vector+'.mat'
    #f_test_y = 'mask_'+date+'.mat'
    if os.path.isfile(file_path+f_test_x):
        mat_dict = loadmat(file_path+f_test_x)
        x_test = np.array(mat_dict['im'])

    y_test_hat = clf.predict(x_test)
    y_test_hat = y_test_hat.reshape((624, -1), order='F')

    savemat(output_path+'y_hat_'+date+'_'+region+'_nn_'+input_vector[1]+'.mat',
        {"y_test_hat" : y_test_hat},
        appendmat=False)
        
    plt.figure(2)
    plt.imshow(y_test_hat, aspect='auto',cmap= colors.ListedColormap(np.array([[0,120,0],[180,100,50]])/255))
    plt.gca().invert_yaxis()
    plt.gca().set_axis_off()
    plt.savefig('/home/akb/Code/PolSAR_ML/output/label_'+date+'_'+region+'_nn.jpg',
                dpi=300,
                bbox_inches='tight')
    plt.show()