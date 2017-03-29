import sys
sys.path.insert(0,'../caffe-master/python')
import caffe
import numpy as np
import scipy.io as scio
import os
import cv2
import time

def getNormMac(net,data):
    net.blobs['data'].data[...] = data
    net.forward()
    code = net.blobs['norm1'].data.copy()
    sz =  code.shape
    code = code.reshape(sz[0],sz[1])
    return code

def processImage(im):
    transformer = caffe.io.Transformer({'data': (1, 3, im.shape[0], im.shape[1])})
    transformer.set_transpose('data', (2, 0, 1))
    mu = np.array([114.79, 114.79, 114.79])
    transformer.set_mean('data', mu)
    transformer.set_raw_scale('data', 255)
    pro_img = transformer.preprocess('data', im)
    return pro_img


def getConvFeatBoxs(net,data):
    net.blobs['data'].data[...] = data
    net.forward()
    code = net.blobs['norm1'].data.copy()
    sz =  code.shape
    code = code.reshape(sz[0],sz[1])
    boxs = net.blobs['boxs'].data.copy()
    sz = boxs.shape
    try:
        boxs = boxs.reshape(sz[2],sz[3])
    except:
        boxs = np.zeros((1,4))
    sz = net.blobs['conv5_3'].data.shape
    return code,boxs,sz

def ranker(qu,data,QE=False):
    qu = np.array(qu)
    sams = []
    name = []
    for i in data:
        sams.append(data[i])
        name.append(i)
    sams = np.array(sams).transpose()
    dist = qu.dot(sams)
    indx = np.argsort(-1 * dist)
    TT = []
    qes = []
    for i in indx:
        TT.append(name[i])
        qes.append(data[name[i]])
    if QE:
        return qes
    return  TT

def QE(qu,data):
    dist = []
    names = []
    for i in data:
        code = data[i]
        mx = -44444
        for q in qu:
            sim = np.sum( q * code) /(np.sqrt( np.sum(q**2))*np.sqrt(np.sum(code**2)))
            if mx < sim:
                mx = sim
        dist.append(mx)
        names.append(i)
    dist = np.array(dist)
    indx = np.argsort(-1*dist)
    rank = []
    for i in indx:
        rank.append(names[i])
    return rank


def queryPositive(queryset,p):
    pos = []
    for j in queryset:
        ff = open(p+'/'+j)
        li = ff.readline()
        while li:
            pos.append(li.strip())
            li = ff.readline()
        ff.close()
    return pos


def rerank(ranks,mp,qu,QE=False,global_mp=None):
    names = []
    dist= []
    qe = []
    for i in ranks:
        regs = None
        if i in mp:
            regs = mp[i]['codes']
        else:
            regs = [global_mp[i]]
        mx = -9999
        cc = None
        for j in regs:
            t = np.sum(j*qu)  /(np.sqrt(np.sum(j**2))*np.sqrt(np.sum(qu**2)))
            if mx < t:
                mx = t
                cc = j
        qe.append(cc)
        names.append(i)
        dist.append(mx)
    dist = np.array(dist)
    indx = np.argsort(-1*dist)
    kk = []
    uu = []
    ll= []
    for i in indx:
        kk.append(names[i])
        ll.append(qe[i])
    if QE:
        return  ll
    return kk,uu


def LQE(ranks,mp,qu,global_mp=None):
    names = []
    dist= []
    # bos = []
    qe = []
    for i in ranks:
        #regs = mp[i]
        # regs = mp[i]['feats']
        regs = None
        if i in mp:
            # boxs = mp[i]['boxs']
            regs = mp[i]['codes']
        else:
            regs = [global_mp[i]]
        mx = -9999
        sm = 0
        bb = None
        cc = None
        for j in regs:
            for q in qu:
                t = np.sum(j*q)/(np.sqrt(np.sum(j**2))*np.sqrt(np.sum(q**2)))
                if mx < t:
                    mx = t
                    cc = j
        qe.append(cc)
        names.append(i)
        dist.append(mx)
    dist = np.array(dist)
    indx = np.argsort(-1*dist)
    kk = []

    for i in indx:
        kk.append(names[i])
    return kk

def globalFeatApply_LW(pdata):
    dataFile = 'data/Lw_MAC_vgg.mat'
    data = scio.loadmat(dataFile)
    means = data['m']
    P = data['P']
    mp =np.load(pdata)
    mp = np.ndarray.tolist(mp['feats'])
    for i in mp:
        j = mp[i].reshape(512,1) - means
        l = P.dot(j)
        ln=np.sqrt(np.sum(l**2))
        l /= ln
        mp[i] = l
    return mp

def querySet(pat):
    files = os.listdir(pat)
    query = []
    for i in files:
        if i.find('query')>=0:
            query.append(i)
    ans = {}
    for i in  query:
        for j in files:
            if j.find(i[:-len('query')-len('.txt')])>=0  and j.find('query') < 0 and j.find('junk')<0:
                if i not in ans:
                    ans[i] = []
                    ans[i].append(j)
                else:
                    ans[i].append(j)
    return ans

def readBox(i):
    f = open(i)
    line = f.readline()
    line = line.strip()
    strs = line.split(' ')
    x1 = float(strs[1])
    y1 = float(strs[2])
    x2 = float(strs[3])
    y2 = float(strs[4])
    return [x1,y1,x2,y2]

def queryRegLWNormMac(gt_path,im_path,dataset='ox'):
    model_def = 'prototxts/test_one.pt'
    model_weight='models/vgg16_faster_reshape.caffemodel'
    net_feat = caffe.Net(model_def,model_weight,caffe.TEST)
    imgs = querySet(gt_path)
    dataFile = 'data/Lw_MAC_vgg.mat'
    data = scio.loadmat(dataFile)
    mean = data['m']
    P = data['P']
    feats = {}
    for i in imgs:
        f = open(gt_path+'/'+i)
        strs = f.readline().strip().split()
        img_path = None
        if dataset == 'ox':
            img_path=strs[0][5:]+'.jpg'
        else:
            img_path=strs[0] + '.jpg'
        img = cv2.imread(im_path+'/'+img_path,1)
        box = readBox(gt_path+'/'+i)
        img_reg = img[box[1]:box[3],box[0]:box[2]]
        cv2.imwrite("tmp.jpg", img_reg)
        im = caffe.io.load_image('tmp.jpg')
        pro_img = processImage(im)
        net_feat.blobs['data'].reshape(1, 3, im.shape[0], im.shape[1])
        feat = getNormMac(net_feat, pro_img).reshape(512,1)
        feat -= mean
        X = P.dot(feat)
        l = np.sqrt(np.sum(X**2))
        X /= l
        feats[i]=X
    if os.path.exists('tmp.jpg'):
        os.remove('tmp.jpg')
    return  feats

def createNet(model_path,proto,gpu_id=0):
    model_weights = model_path
    if gpu_id >= 0 :
        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)
    else:
        caffe.set_mode_cpu()
    net = caffe.Net(proto,model_weights,caffe.TEST)
    return net

def getHashCode(hash_net,data1,mm=None):
    net = hash_net
    s = []
    if isinstance(data1,dict):
        for i in data1:
            s.append(data1[i].copy())
        s = np.array(s).reshape(len(data1),512)
    else:
        s = data1
    if mm is not None:
        s -= mm
    out = net.forward_all(data=s)
    feats = out['B']
    if isinstance(data1,dict) == False:
        return feats
    mp= {}
    for i,j in zip(data1,feats):
        mp[i] = j
    return mp


def getGlobalFeatMean(mp):
    a = None
    for i in mp:
        if a == None:
            a = mp[i]
        else:
            a += mp[i]
    return  a/len(mp)


def calc_map(positive,rank):
    ap = 0.0
    shot = 0.0
    n = 0.
    old = 0.0
    old_pre = 1.
    recall = 0.
    pre = 0.
    for i in rank:
        try:
            _= positive.index(i[:-len('.jpg')])
            shot += 1.
            recall = shot*1.0 / len(positive)
            pre = shot/(n+1.0)
            ap += (recall - old) *((old_pre + pre)/2.)
            n+= 1
            old = recall
            old_pre = pre
        except:
            n+= 1.
    return ap


def calc_feats_time(qu, data, QE=False):
    sams = []
    name = []
    for i in data:
        sams.append(data[i])
        name.append(i)
    sams = np.array(sams).transpose()
    s = time.clock()
    dist = (qu.transpose().dot(sams))[0]
    tt = np.sqrt(np.sum(qu**2))
    ttt = np.sqrt( np.sum(sams**2,axis=1) )
    tt *= ttt
    dist /= tt
    indx = (np.argsort(-1*dist))[0]
    e = time.clock()
    TP = []
    for i in indx:
       TP.append(name[i])
    return e - s,TP

def calc_hash_time(qu, data, QE=False):

    qu = np.array(qu)
    sams = []
    name = []
    for i in data :
        sams.append(data[i])
        name.append(i)
    sams = np.array(sams).transpose()
    s=time.clock()
    dist = qu.dot(sams)
    _ = np.argsort(-1*dist)
    e = time.clock()
    TT = []
    for i in _:
        TT.append(name[i])
    return e - s,TT


def cal_ap(dataset='ox'):
    ranks = os.listdir('ranks')
    if os.path.exists('ans.txt'):
        os.remove('ans.txt')
    for i in ranks:
        f = open('ranks/'+i,'r')
        l = f.readline()
        ss = set()
        while l:
            ss.add(l.strip())
            l = f.readline()
        f.close()
        if dataset=='ox':
            if len(ss)!=5063 and len(ss)!=105079:
                raise NameError('rank_list file is not correct !')
        if dataset == 'paris':
            if len(ss) != 6392 and len(ss)!= 106408:
                raise NameError('rank_list file is not correct !')
    for i in ranks:
        j = i[:-len('_query.txt')]
        cmd = './compute_ap ../data/drh_data/'+dataset+'_gt/' + j + '  ranks/' + i + '>> ans.txt'
        os.system(cmd)
    f = open('ans.txt', 'rb')
    h = f.readline()
    t = 0
    p = 0
    while h:
        t += float(h)
        h = f.readline()
        p += 1.
    f.close()
    if os.path.exists('ans.txt'):
        os.remove('ans.txt')
    return  t / p

def compactBit(mp):
    ret = []
    for i in mp:
        code = mp[i]
        c = []
        h = 0
        for ii,j in enumerate(code):
            if ii%8==0 and ii !=0 :
                   c.append(np.uint8(h))
                   h = 0
            if j == 1:
                h += (1<<((ii)%8))
        c.append(np.uint8(h))
        # print c
        # raw_input()
        ret.append(c)
    return np.array(ret)
