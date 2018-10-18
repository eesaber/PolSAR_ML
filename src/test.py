# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Created on Wed Oct 10 15:29:30 2018

@author: Usted
"""
import matplotlib.pyplot as plt
import numpy as np
import os
'''
import tensorflow as tf
size = 500
W = tf.random_normal([size, size], name='W')
X = tf.random_normal([size, size], name='X')
mul = tf.matmul(W, X, name='mul')
sum_result = tf.reduce_sum(mul, name='sum')


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.380)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) 
result = sess.run(sum_result)
'''
dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)
print(dir_path[0:dir_path.find('src')])
