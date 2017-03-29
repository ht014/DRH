#!/usr/bin/env python
# encoding: utf-8
import scipy.io as scio
import numpy as np
import sys
sys.path.insert(0,'../caffe-master/python')
import caffe
from utils import *
import os



###need to config arguments###
dataset= 'paris'
save_feat_names = dataset+'_global.npz'
##############################

caffe.set_mode_gpu()
caffe.set_device(0)
model_def = 'prototxts/test_one.pt'
model_weight='models/vgg16_faster_reshape.caffemodel'
net = caffe.Net(model_def,model_weight,caffe.TEST)
net = caffe.Net(model_def,model_weight,caffe.TEST)
rt_path='../data/drh_data/'

imgs = os.listdir(rt_path+dataset)
feats = {}
indx = -1
for i in imgs:
     indx+=1
     im = None
     try:
          im = caffe.io.load_image(rt_path +dataset+'/' + i)
          # im = caffe.io.load_image(i)
     except:
          continue
     net.blobs['data'].reshape(1,3,im.shape[0],im.shape[1])
     pro_img = processImage(im)
     feat = getNormMac(net,pro_img)
     feats[i] = feat
     print indx,i
np.savez('outputs/'+save_feat_names,feats = feats)
print save_feat_names,'save done!'
