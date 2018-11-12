# -*- coding: utf-8 -*-
#!/usr/bin/env python3
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import os.path

from keras import optimizers, Sequential, utils
from keras.layers import Dense

#%% read image
work_path = os.path.dirname(os.path.realpath(__file__))
work_path = work_path[0:work_path.find('src')]
file_path = work_path+'data/final_version/'
model_path =   work_path+'model/'

#f_name_x = 'image_070426_3_(4).mat'
#f_name_y = 'mask_070426_3.mat'
f_name_x = 'image_090811_(1).mat'
f_name_y = 'mask_090811.mat'
#%%
if os.path.isfile(file_path+f_name_x):
    mat_dict = loadmat(file_path+f_name_x)
    x = np.array(mat_dict['im'])
if os.path.isfile(file_path+f_name_y):
    mat_dict = loadmat(file_path+f_name_y)
    y = np.array(mat_dict['gt'])

#%% reshape label
labels = 2
y_train = y.reshape((-1, 1), order='F').squeeze().astype('float32')


#%% NN 
batch_size = 200
epochs = 20
model = Sequential()
model.add(Dense(20, input_shape=(x.shape[1],), activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='softmax'))
optimizer = optimizers.SGD(lr=0.01, momentum=0.9, decay=0.001,nesterov=False)
model.compile(optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy'])
model.fit(x, y_train,
    validation_split=0.1,
    epochs=epochs,
    batch_size=batch_size,
    verbose=1,
    class_weight={0:1.0, 1:5.0},
    shuffle=True)

y_hat = model.predict(x, verbose=1)
y_hat = y_hat.reshape((624, -1), order='F')
# 
save_results = False
if save_results:
    savemat(file_path+'y_hat_090811.mat',
        {"y_hat" : y_hat},
        appendmat=False)

#%% Show the confusion matrix
Mm = np.sum((y_hat==1)*(y==1))/y.size
Mf = np.sum((y_hat==1)*(y==0))/y.size
Fm = np.sum((y_hat==0)*(y==1))/y.size
Ff = np.sum((y_hat==0)*(y==0))/y.size

print("Total Accuracy: %f" % (np.sum(y==y_hat)/y_hat.size))
print("Confusion matrix:\n pred/true: MYI  FYI\n MYI     %.3f %.3f\n FYI     %.3f %.3f \n"
 % (Mm, Mf, Fm, Ff))
#%%
plt.figure(1)
plt.imshow(y_hat, aspect='auto',cmap= colors.ListedColormap(np.array([[0,120,0],[180,100,50]])/255))
plt.gca().invert_yaxis()
plt.gca().get_xaxis().set_ticks([])
plt.gca().get_yaxis().set_ticks([])
plt.gca().set_axis_off()
plt.show()
plt.savefig('/home/akb/Code/PolSAR_ML/output/label_070428_NN.jpg',
            dpi=300,
            bbox_inches='tight')
            

