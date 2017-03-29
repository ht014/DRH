#!/usr/bin/env python
# encoding: utf-8
import sys
sys.path.insert(0,'../caffe-master/python')
sys.path.insert(0,'.')
import caffe


caffe.set_mode_gpu()
caffe.set_device(0)
solver = None
solver = caffe.SGDSolver('prototxts/solver.pt')
MAX = 5000
for i in xrange(MAX):
    solver.step(1)

