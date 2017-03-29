#!/usr/bin/env python
# encoding: utf-8
import scipy.io as scio
import numpy as np
import sys
sys.path.insert(0,'../caffe-master/python')
from utils import *
import caffe
import os


##########need to config############
dataset= 'paris'
save_feats_name = dataset+'_slides_0.6.npz'
#####################################


caffe.set_mode_gpu()
caffe.set_device(0)
rt_path='../data/drh_data/'
model_def = 'prototxts/slide_box.pt'
model_weight='models/vgg16_faster_reshape.caffemodel'
net = caffe.Net(model_def,model_weight,caffe.TEST)
imgs = os.listdir(rt_path+dataset)
feats = {}
indx = -1
for i in imgs:
     indx+=1
     im = None

     try:
          im = caffe.io.load_image(rt_path+dataset+'/' + i)
     except:
          continue
     src_shape = im.shape
     print indx, i
     net.blobs['data'].reshape(1, 3, im.shape[0], im.shape[1])
     pro_img = processImage(im)
     feat,box,scales = getConvFeatBoxs(net,pro_img)
     h = src_shape[0]*1./scales[2]
     w = scales[1]*1./scales[3]
     box[:, 0] *= h
     box[:, 1] *= h
     box[:, 2] *= w
     box[:, 3] *= w
     # print feat.shape
     feats[i] ={'feats':feat,'boxs':box}

np.savez('outputs/'+save_feats_name,feats = feats)
print save_feats_name,'save done!'
