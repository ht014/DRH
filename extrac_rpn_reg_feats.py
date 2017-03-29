#!/usr/bin/env python
# encoding: utf-8
import scipy.io as scio
import numpy as np
import sys
sys.path.insert(0,'../caffe-master/python')
import caffe
import cv2
import os
import utils

###need to config parameters####
data_set='ox'
save_rpn_feat_name=data_set+'_rpn.npz'
###########end config###########

caffe.set_mode_gpu()
caffe.set_device(0)
model_def = 'prototxts/rpn.pt'
model_weight='models/vgg16_faster_reshape.caffemodel'
net = caffe.Net(model_def,model_weight,caffe.TEST)
rt= '/home/xing/py-faster-rcnn/data/drh_data/'+data_set+'/'
imgs = os.listdir(rt)
feats = {}
indx = -1
im_id = 0
for i in imgs:
     indx+=1
     im = cv2.imread(rt+i,1)
     try:
          im = caffe.io.load_image(rt + i)
     except:
          continue
     net.blobs['data'].reshape(1, 3, im.shape[0], im.shape[1])
     pro_img = utils.processImage(im)
     info = [im_id,im.shape[0],im.shape[1]]
     net.blobs['info'].data[...] = np.array(info).reshape(1,1,1,3)
     feat = utils.getNormMac(net,pro_img)
     feats[i] = feat
     im_id += 1
     print feat.shape
     print indx,len(feat),i
np.savez('outputs/'+save_rpn_feat_name,feats = feats)
print save_rpn_feat_name,'save done!'
