#!/usr/bin/env python3
# -*- coding: utf-8 -*-  

from FileIO import read

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model

import numpy as np
import matplotlib.pyplot as plt

if __name__=='__main__':
    s = read()