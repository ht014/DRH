import sys
import caffe
import numpy as np
import scipy.io as scio


def getGlobalFeats():
    dataFile = 'data/Lw_MAC_vgg.mat'
    data = scio.loadmat(dataFile)
    means = data['m']
    P = data['P']
    mp = np.load('outputs/ox_global.npz')
    mp = np.ndarray.tolist(mp['feats'])
    pp = []
    for i in mp:
        j = mp[i].reshape(512, 1) - means
        l = P.dot(j)
        ln = np.sqrt(np.sum(l ** 2))
        l /= ln
        pp.append(l[0])

    return np.array(pp)

class LMDBData(caffe.Layer):
    def reshape(self, bottom, top):
        top[0].reshape(self.batch_size,512)

    def setup(self, bottom, top):
        self.batch_size = 128
        # print "created succeed"
        self.cur = 0
        feats = getGlobalFeats()

        self.alldata = feats
        m = np.mean(self.alldata,axis=0)
        self.alldata -= m

    def forward(self, bottom, top):
        tmp =  self.getNetBatch()
        top[0].data[...] = tmp


    def backward(self, top, propagate_down, bottom):
        pass

    def getNetBatch(self):
        if self.cur+self.batch_size > len(self.alldata):
            np.random.shuffle(self.alldata)
            self.cur = 0
        t = self.alldata[self.cur: self.cur+self.batch_size]
        self.cur += self.batch_size
        return  t

