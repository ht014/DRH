import sys
import caffe
import numpy as np

class DHLoss(caffe.Layer):
    def reshape(self, bottom, top):
        top[0].reshape(1)
        self.diff = np.zeros_like(bottom[0].data,dtype=np.float32)

    def setup(self, bottom, top):
        self.lamd1 = 100.

    def forward(self, bottom, top):
        self.diff[...] = bottom[0].data-bottom[1].data-self.lamd1*bottom[1].data
        top[0].data[...] = np.sum((bottom[0].data-bottom[1].data)**2-self.lamd1*(bottom[1].data**2)) / bottom[0].num

    def backward(self, top, propagate_down, bottom):
        bottom[0].diff[...] = self.diff / bottom[0].num

