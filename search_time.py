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
data_set_name='paris'
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
imgs = utils.querySet(gt_path)
if os.path.exists('ranks'):
    shutil.rmtree('ranks')
    os.mkdir('ranks')
ans = 0
n = 0
for i in imgs:
    f = open(gt_path+'/'+i)
    strs = f.readline().strip().split()
    img_path = None
    if data_set_name == 'ox':
        img_path=strs[0][5:]+'.jpg'
    if data_set_name =='paris':
        img_path=strs[0]+'.jpg'
    n += 1
    qu = query_codes[i]
    rank = None
    if IS_HASH:
        tm,rank = calc_hash_time(qu, mp)
        ans += tm
        print i,'costs %fs' %tm
    else:
        tm , rank = calc_feats_time(qu, mp)
        ans += tm
        print i, 'costs  %fs'%tm
    f = open("ranks/" + i, 'wb')
    for j in rank:
        f.write(j[:-4] + '\n')
    f.close()

print 'mAP=',utils.cal_ap(data_set_name)
print 'alll querys average costs  %fs'%(ans / n)



