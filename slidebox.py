#!/usr/bin/env python
# encoding: utf-8

import caffe
import numpy as np

def slideWindow(h,w):
    #shape = img.shape
    #h = shape[0]
    #w = shape[1]
    if h>w:
         hs = [h, 0.5*h ,0.333*h]
         ws = [w, 0.5*w ]
    else:
        hs = [h, 0.5*h]
        ws = [w, 0.5*w , 0.333*w]
    overlap = 0.6
    xy = []
    for hi in hs:
        for wi in ws:
            x = y = 0.
            if max(wi,hi)*1./min(wi,hi) >= 3:
                continue
            x_step = (1. - overlap)*wi
            y_step = (1. - overlap)*hi
            while y+hi<=h:
                x = 0.
                while x+wi <=w:
                     xy.append([y,y+hi-1,x,x+wi-1])
                     x += x_step
                y += y_step
    return xy
class RegMacLayer(caffe.Layer):
    def setup(self,bottom,top):
        pass
    def reshape(self,bottom,top):
        top[0].reshape(1,512,5,5)
        top[1].reshape(1,23,1,1)
        pass
    def forward(self,bottom,top):
        conv5 = bottom[0].data[...] #1x512x34x40
        h = conv5.shape[-2]
        w = conv5.shape[-1]
        boxs = slideWindow(h,w)
        sz = conv5.shape
        # print 'conv5 shape:',sz,'len boxs:',len(boxs)
        if len(boxs) >=1:
            tmp = np.zeros((len(boxs),512))
            for indx,i in enumerate(boxs):
                for j in xrange(sz[1]):
                    tmp[indx][j] = np.max(conv5[0,j,i[0]:i[1],i[2]:i[3]])
            tmp = tmp.reshape(len(boxs),512,1,1)
            top[0].reshape(len(boxs),512,1,1)
            top[0].data[...] = tmp
            top[1].reshape(1,1,len(boxs),4)
            n_boxs = np.array(boxs).reshape(1,1,len(boxs),4)
            top[1].data[...] = n_boxs
        else:
            top[0].reshape(1,512,1,1)
            top[0].data[...]=np.zeros((1,512,1,1))
            top[1].data[...] = np.zeros((1,23,1,1))

    def backward(self,bottom,top,propagate_down):
        pass
