import sys
import caffe
import numpy as np

class Sgn(caffe.Layer):
    def reshape(self, bottom, top):
        top[0].reshape(bottom[0].data.shape[0],bottom[0].data.shape[1])
        self.diff = np.zeros_like(bottom[0].data)
        pass
    def setup(self, bottom, top):
        pass
    def forward(self, bottom, top):
        top[0].data[np.where(bottom[0].data < 0)] = -1
        top[0].data[np.where(bottom[0].data>=0)] = 1
        # print top[0].data[0]

    def backward(self, top, propagate_down, bottom):
        self.diff[...] = top[0].diff[...]
        pass
