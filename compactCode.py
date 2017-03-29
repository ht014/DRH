#!/usr/bin/env python
# encoding: utf-8
import scipy.io as scio
import numpy as np
import utils
import sys
sys.path.insert(0,'../caffe-master/python')
sys.path.insert(0,'.')
import caffe
import shutil
from utils import *


###########         config params      ############
data_set_name='ox'
data_flickr = 'flickrs_global.npz'
IS_ADD_FLICKRS=True
IS_HASH=True
####################################################
caffe.set_mode_gpu()
caffe.set_device(0)
hash_net=createNet('models/dh_mac_1024_'+data_set_name+'.caffemodel',"prototxts/test_.pt",0)
gt_path='/home/xing/py-faster-rcnn/data/drh_data/'+data_set_name+'_gt'
img_path='../data/drh_data/'+data_set_name
global_feats_path= 'outputs/'+data_set_name+'_global.npz'
mp = globalFeatApply_LW(global_feats_path)
flickr = None
if IS_ADD_FLICKRS:
    flickr=globalFeatApply_LW('outputs/'+data_flickr)
    mp = dict(mp, **flickr)
means = getGlobalFeatMean(mp).transpose()[0]
query_codes = queryRegLWNormMac(gt_path,img_path,dataset=data_set_name)
if IS_HASH:
    mp = getHashCode(hash_net,mp,means)
    query_codes = getHashCode(hash_net,query_codes,means)

B = compactBit(mp).transpose()
scio.savemat('outputs/'+data_set_name+'_compact_code.mat', {'B': B})