# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import os
import PIL
import numpy as np

def readslc(filename, numel=-1, offset=0):
    with open(filename, 'r') as f:
        f.seek(offset, os.SEEK_SET)
        # count: number of items to read. -1 means all items (i.e., the complete file).
        x = np.fromfile(f,dtype='<f4',count=numel)
        x = x[0:2:-1, :]*np.exp(x[1:2:-1, :]*1j)
    return x

def pile(nums, size, patch_size):
    nums = np.reshape(nums, size)
    '''
    im = Image.open(input)
    imgwidth, imgheight = im.size
    for i in range(0,imgheight,height):
        for j in range(0,imgwidth,width):
            box = (j, i, j+width, i+height)
            a = im.crop(box)
            try:
                o = a.crop(area)
                o.save(os.path.join(path,"PNG","%s" % page,"IMG-%s.png" % k))
            except:
                pass
            k +=1
    '''

if __name__ == '__main__':
    readslc('1')