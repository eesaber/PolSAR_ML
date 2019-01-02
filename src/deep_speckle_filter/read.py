# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import os
import PIL
import numpy as np
import matplotlib.pyplot as plt

def readslc(filename, numel=-1, offset=0):
    with open(filename, 'r') as f:
        f.seek(offset, os.SEEK_SET)
        # count: number of items to read. -1 means all items (i.e., the complete file).
        x = np.fromfile(f,dtype='<f4',count=numel)
        x = x[0::2]*np.exp(x[1::2]*1j)
    return x

def pile(im, patch_size):
    # im: 3d array, im[i,j,k]
    # i: row number, j: column number, k: channel
    # piles: 4d array, im[l,i,j,k]
    # l: number of subimage
    l = 0
    imgheight, imgwidth  = patch_size
    piles = np.zeros((np.floor(np.prod(im.shape[0:2]/patch_size)), im.shape))
    for i in range(0,imgheight, im.shape[0]):
        for j in range(0,imgwidth, im.shape[1]):
            piles[l,:,:,:] = im[i:i+imgheight, j:j+imgwidth,:]
            l+=1

    return piles

if __name__ == '__main__':
    # image size parameters
    SIZE = np.array([95000,10000])
    numel = 2*np.prod(SIZE)
    OFFSET = 2*9800*95000*4
    FILE = '/home/akb/下載/Haywrd_23501_18039_014_180801_L090HH_CX_01.slc'
    # varaible
    sig = {}
    sig['hh'] = np.reshape(readslc(FILE, numel=numel, offset=OFFSET), SIZE)
    # sig['hv'] = readslc('1')
    # sig['vv'] = readslc('1')
    
    plt.figure(1)
    plt.imshow(10*np.log10(abs(sig['hh'])), aspect='auto', cmap=plt.get_cmap('jet'))
    plt.clim([-20, 20])
    plt.gca().set_axis_off()
    plt.savefig('/home/akb/Code/PolSAR_ML/output/ss.jpg',
            dpi=300,
            bbox_inches='tight')
    