# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import scipy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import h5py
import os.path

from sklearn.neural_network import MLPClassifier

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import Dropout, Activation, Flatten
from keras.models import Model, Sequential
from keras import backend as K
from keras.preprocessing import image


#%% tensorflow setting
if 'tensorflow' == K.backend():
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    config.gpu_options.visible_device_list = "0"
    set_session(tf.Session(config=config))

#%% read image
path = '/home/akb/Code/PolSAR_ML/data/'
if os.path.isfile(path+'image.mat'):
    mat_dict = scipy.io.loadmat(path+'image')
    raw = np.array(mat_dict['image'])
if os.path.isfile(path+'label.mat'):
    mat_dict = scipy.io.loadmat(path+'label')
    label = np.array(mat_dict['bw'])
#%%
#label = read('label.mat')
plt.figure(1)
plt.imshow(10*np.log10(raw[:,:,0]))
#plt.gca().invert_yaxis()    
plt.show()

#%% reshape image and label
x = raw.reshape((-1, raw.shape[2]), order='F')
y = label.reshape((-1, 1), order='F').squeeze()
x_train = x[1:2000000,:]
x_test = x[2000001:-1,:]
y_train = y[1:2000000]
y_test = y[2000001:-1]
'''
# recale 
x_min = np.amin(x, axis=0)
x_max = np.amax(x, axis=0)
u = 0
l = 1
x = (x-x_min)/(x_max-x_min)*(u-l) + l
'''
#%% NN 
clf = MLPClassifier(solver='lbfgs', alpha=1e-8, activation='relu',
                hidden_layer_sizes=(10,30,30,20), random_state=1, 
                verbose=True)
clf.fit(x, y)

y_hat = clf.predict(x)
y_hat = y_hat.reshape(raw.shape[0:2],order='F')
plt.figure(1)
plt.imshow(y_hat)
plt.gca().invert_yaxis()    
plt.show()
print(np.sum(label!=y_hat)/y_hat.size)

#%% Data augmentation
datagen = image.ImageDataGenerator(samplewise_center=True,
    rotation_range=45,
    horizontal_flip=True,
    vertical_flip=True)

load4dir = '/home/akb/Code/PolSAR_ML/data/training_data/hands1.jpg'
save2dir = '/home/akb/Code/PolSAR_ML/data/generated_data/'

x = image.img_to_array(image.load_img(load4dir))
x1 = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 267, 400, 3)

seed = 1
i = 0
for batch in datagen.flow(x1, batch_size=1,seed=seed,
                          save_to_dir=save2dir, save_format='jpg'):
    i += 1
    if i > 9:
        break  # otherwise the generator would loop indefinitely
        
i = 0
#%%
i = 1
plt.figure()
for f in os.listdir(save2dir):
    plt.subplot(2,5,i)
    plt.imshow(mpimg.imread(save2dir+f))
    i=i+1
plt.show()

#%% Training CNN        

seg_cnn = Sequential()
seg_cnn.add(Conv2D(32, (3, 3), activation='relu', padding='same',
                 input_shape=x_train.shape[1:]))
seg_cnn.add(MaxPooling2D((2, 2), padding='same'))
seg_cnn.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
seg_cnn.add(MaxPooling2D((2, 2), padding='same'))
seg_cnn.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
seg_cnn.add(MaxPooling2D((2, 2), padding='same'))

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

seg_cnn.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
seg_cnn.add(UpSampling2D((2, 2)))
seg_cnn.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
seg_cnn.add(UpSampling2D((2, 2)))
seg_cnn.add(Conv2D(16, (3, 3), activation='relu'))
seg_cnn.add(UpSampling2D((2, 2)))
seg_cnn.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))
seg_cnn.add(Activation('softmax'))

seg_cnn.compile(optimizer='adadelta', loss='binary_crossentropy') 
seg_cnn.fit(x_train, x_train,
                epochs=25,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test))