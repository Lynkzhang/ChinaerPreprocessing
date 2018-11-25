#!/usr/bin/env python

from __future__ import print_function
import argparse
import multiprocessing
import random
import sys

import numpy as np
import cupy
import chainer
import chainer.cuda
from chainer import training
from chainer import backend
from chainer.training import extensions

import chainermn

# Check Python version if it supports multiprocessing.set_start_method,
# which was introduced in Python 3.4
major, minor, _, _, _ = sys.version_info
if major <= 2 or (major == 3 and minor < 4):
    sys.stderr.write("Error: ImageNet example uses "
                     "chainer.iterators.MultiprocessIterator, "
                     "which works only with Python >= 3.4. \n"
                     "For more details, see "
                     "http://chainermn.readthedocs.io/en/master/"
                     "tutorial/tips_faqs.html#using-multiprocessiterator\n")
    exit(-1)

#imagenet
rgbmean=np.array([123.68, 116.779, 103.939])
targetsize = 256.0

class PreprocessedDataset(chainer.dataset.DatasetMixin):

    def __init__(self, path, root, mean, crop_size, random=True):
        self.base = chainer.datasets.LabeledImageDataset(path, root)
        self.mean = mean.astype('f')
        self.crop_size = crop_size
        self.random = random

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        # It reads the i-th image/label pair and return a preprocessed image.
        # It applies following preprocesses:
        #     - Cropping (random or center rectangular)
        #     - Random flip
        #     - Scaling to [0, 1] value
        crop_size = self.crop_size

        image, label = self.base[i]
        ci, h, w = image.shape
        #print("original ci: %d"%ci)
        #print(image.shape) 
        #targetsize = 256.0
        shortside=min(h,w)
        ratio = targetsize * 1.0 / shortside
        #print(ratio)
        n_h = int(ratio * h)
        n_w = int(ratio * w)
        image=chainer.functions.reshape(image,(1,ci,h,w))
        image=chainer.functions.resize_images(image,(n_h,n_w))
        image=chainer.functions.reshape(image,(ci,n_h,n_w))
        #print("after: "%ci)
        #print(image.shape)


        if self.random:
            # Randomly crop a region and flip the image
            top = random.randint(0, n_h - crop_size - 1)
            left = random.randint(0, n_w - crop_size - 1)
            if random.randint(0, 1):
                image = image[:, :, ::-1]
        else:
            # Crop the center
            top = (n_h - crop_size) // 2
            left = (n_w - crop_size) // 2
        bottom = top + crop_size
        right = left + crop_size

        image = image[:, top:bottom, left:right]
       # print("image cropped shape:")
       # print(image.shape)
        image=chainer.functions.transpose(image,axes=(1,2,0))
       # print(image.shape)
       # rgbmean=np.array([123.68, 116.779, 103.939])
       # for i in range(ci):
        image -= rgbmean
        image=chainer.functions.transpose(image,axes=(2,0,1)) 
       # print(type(image))
        #image -= self.mean[:, top:bottom, left:right]
        image *= (1.0 / 255.0)  # Scale to [0, 1]
        #print(image.dtype,image.shape)
        #print(type(image.array))
        return image.array, label
