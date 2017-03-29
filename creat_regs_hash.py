#!/usr/bin/env python
# encoding: utf-8
import scipy.io as scio
import numpy as np
import utils
import cv2

data_set='paris'
feats_name ='outputs/'+data_set+'_slides_0.6.npz'
# feats_name ='outputs/ox_rpn.npz'
save_code_name = data_set+"_slides_hashcode_0.6.npz"

mp_reg = np.ndarray.tolist(np.load(feats_name)['feats'])
dataFile = 'data/Lw_MAC_vgg.mat'
data = scio.loadmat(dataFile)
mean = data['m'].transpose()[0]
P = data['P']
net = utils.createNet('models/dh_mac_1024_iter_1_0.748_ox.caffemodel',"prototxts/test_.pt",0)
cnt = 0
mp_new ={}
# mm=utils.getMean1(mp_reg,'/home/xing/py-faster-rcnn/data/drh_data/'+'ox_gt','../data/drh_data/'+'ox_builds')
# np.savez('outputs/mean.npz',mean=mm)
for i in mp_reg:
    data = mp_reg[i]['feats']
    box = mp_reg[i]['boxs']
    data = np.array(data)
    data -= mean
    data = P.dot(data.transpose())
    l = np.sqrt(np.sum(data**2,axis=0))
    data /=l
    data = data.transpose()
    # data -= mm
    out = net.forward_all(data=data)
    codes = out['B'].copy()
    print i,cnt,codes.shape
    cnt +=1
    #mp_new[i]= {"codes":codes}
    mp_new[i]= {"codes":codes,"boxs":box}

np.savez('outputs/'+save_code_name,regs=mp_new)
print save_code_name,'saved done!'
