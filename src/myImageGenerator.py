# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import os
from keras.preprocessing import image
from scipy.io import loadmat, savemat
from skimage.transform import resize

def myImageGenerator():
    #set path variable
    work_path = os.path.dirname(os.path.realpath(__file__))
    work_path = work_path[0:work_path.find('src')]
    file_path = work_path+'data/'
    augmentation_file_path = work_path+'data_aug/'
    # read file
    mat_dict = loadmat(file_path+'image.mat')
    x_train = np.array(mat_dict['x_train'])
    x_train = np.expand_dims(x_train, axis=0)
    mat_dict = loadmat(file_path+'mask.mat')
    y_train = np.array(mat_dict['y_train'])
    y_train = np.expand_dims(np.expand_dims(y_train, axis=-1), axis=0)

    if 0:
        plt.figure()
        plt.imshow(y_train, aspect='auto',cmap='gray')
        plt.gca().invert_yaxis()
        plt.gca().set_axis_off()
        plt.show()

    # Reference: https://github.com/keras-team/keras/issues/3059
    # Declare the same arguments and the same random seed for ImageDataGenerator
    seed = 87
    data_gen_args = dict(
        data_format="channels_last",
        samplewise_center=False,
        #horizontal_flip=True,
        vertical_flip=True,
        width_shift_range=0.3,
        height_shift_range=0.2,
        shear_range=0.5,
        zoom_range=0.5,
        rotation_range=20,
        fill_mode='constant',
        cval=0)
        
    data_flow_args = dict(
        save_to_dir=augmentation_file_path,
        batch_size=10,
        save_format='jpeg',
        seed=seed)
    # Pass the same seed and keyword arguments to each ImageDataGenerator
    image_datagen = image.ImageDataGenerator(**data_gen_args)
    mask_datagen = image.ImageDataGenerator(**data_gen_args)

    image_generator = image_datagen.flow(
        x_train,
        save_prefix='image',
        **data_flow_args)
    mask_generator = mask_datagen.flow(
        y_train,
        save_prefix='mask',
        **data_flow_args)
    #combine generators into one which yields image and masks
    train_generator = zip(image_generator, mask_generator)

    # Image augmentation
    batches = 0
    data_set_size = 100
    img_h = 96
    img_w = 496
    channels = 3
    new_Xtrain = np.zeros((data_set_size,img_h,img_w,channels))
    new_Ytrain = np.zeros((data_set_size,img_h,img_w))
    for x_batch, y_batch in train_generator:
        #print(x_batch.shape)
        #new_Xtrain[batches,:,:,:] = resize(np.squeeze(x_batch, axis=0), (img_h, img_w, channels), anti_aliasing=True)
        #new_Ytrain[batches,:,:] = resize(np.squeeze(y_batch, axis=(0, -1)), (img_h, img_w), anti_aliasing=True)
        batches += 1
        if batches >= data_set_size :
            break
    
    # Save image
    if 0:
        savemat(augmentation_file_path+'x_train.mat',
            {'x_train': new_Xtrain.astype(np.float32)})
        savemat(augmentation_file_path+'y_train.mat',
            {'y_train': new_Ytrain.astype(np.uint8)})

    return (new_Xtrain, new_Ytrain)

if __name__ == '__main__':
    myImageGenerator()
