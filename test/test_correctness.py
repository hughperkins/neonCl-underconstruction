# Copyright 2016 Hugh Perkins, All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import numpy as np
import time
import pyopencl as cl
from winogradcl import api

#from winogradcl.backends.kernels.cl.mycltensor import MyClTensor
#from winogradcl.layers.layer import Convolution
#from winogradcl.backends.make_backend import make_backend

its = 1

mf = cl.mem_flags

def printDims(W, I):
    Ci = W.shape[0]
    iH = I.shape[1]
    iW = I.shape[2]
    Co = W.shape[3]
    kH = W.shape[1]
    kW = W.shape[2]
    print('Ci', Ci, 'iH', iH, 'iW', iW, 'Co', Co, 'kH', kH, 'kW', kW)

def check_gradWeights(O, I, W, gradO, gradW, ci, h, w, co, eps=1e-2):
#    eps = 1e4 #hack
    N = I.shape[3]
    iH = I.shape[1]
    iW = I.shape[2]
    Ci = W.shape[0]
    kH = W.shape[1]
    kW = W.shape[2]
    Co = W.shape[3]
    oH = iH # assuming padded, which it is
    oW = iW # assuming padded, which it is
#    print('Ci', Ci, 'iH', iH, 'iW', iW, 'Co', Co, 'kH', kH, 'kW', kW)

#    ih = h
#    iw = w
    kh = h
    kw = w
#    ci = c

    padw = 1
    padh = 1

    sum = 0

    for ow in range(oW):
        for oh in range(oH):
            ih = oh + kh - padh
            iw = ow + kw - padw
            for n in range(N):
                if ih >= 0 and iw >= 0 and ih < iH and iw < iW:
                    v = I[ci, ih, iw, n] * gradO[co * iH * iW + oh * iW + ow, n]
                    sum += v
    cpu_value = sum
    gpu_value = gradW[ci, kh, kw, co]
    print('checkGradW gpu', gpu_value, 'cpu', cpu_value)
    assert abs(cpu_value - gpu_value) < eps

def check_gradI(O, I, W, gradO, gradI, c, h, w, n, eps=1e-4):
    N = I.shape[3]
    iH = I.shape[1]
    iW = I.shape[2]
    Ci = W.shape[0]
    kH = W.shape[1]
    kW = W.shape[2]
    Co = W.shape[3]
    oH = iH # assuming padded, which it is
    oW = iW # assuming padded, which it is
#    print('Ci', Ci, 'iH', iH, 'iW', iW, 'Co', Co, 'kH', kH, 'kW', kW)

    ih = h
    iw = w
    ci = c

    padw = 1
    padh = 1

    sum = 0
    for co in range(Co):
        for kh in range(kH):
            for kw in range(kW):
                ow = iw - kw + padw
                oh = ih - kh + padh
                if ow >= 0 and oh >= 0 and ow < oW and oh < oH:
                    v = gradO[co * iH * iW + oh * iW + ow, n] * W[ci, kh, kw, co]
                    sum += v
    cpu_value = sum
    gpu_value = gradI[c, ih, iw, n]
    print('checkGradI gpu', gpu_value, 'cpu', cpu_value)
    assert abs(cpu_value - gpu_value) < eps

def checkO(O, W, I, c, h, w, n, eps=1e-4):
    Ci = W.shape[0]
    iH = I.shape[1]
    iW = I.shape[2]
    Co = W.shape[3]
    kH = W.shape[1]
    kW = W.shape[2]
#    print('Ci', Ci, 'iH', iH, 'iW', iW, 'Co', Co, 'kH', kH, 'kW', kW)

    co = c
    padw = 1
    padh = 1

    # we are going to apply entire kernel, over all input channels, to the input
    # image, in one location
    sum = 0
    for kw in range(kW):
        for kh in range(kH):
            for ci in range(Ci):
                ih = h + kh - padh
                iw = w + kw - padw
                if ih >= 0 and iw >= 0 and ih < iH and iw < iW:
                    v = I[ci, ih, iw, n] * W[ci, kh, kw, co]
                    sum += v
    cpu_value = sum
    gpu_value = O[c*iH*iW + h*iW + w,n]
    print('checkO c', c, 'h', h, 'w', w, 'n', n, 'cpu %.6f gpu %.6f' % (sum, gpu_value))
    assert abs(sum - gpu_value) < eps
    return ""

class MyTensor(object):
    def __init__(self, gpudata, shape, size):
        self.gpudata = gpudata
        self.size = size
        self.dtype = np.float32
        self.cpudata = None
        self.shape = shape

    @staticmethod
    def from_np(np_data):
        cudabuf = cuda.mem_alloc(np_data.nbytes)
        cuda.memcpy_htod(cudabuf, np_data)
#        self.cpudata = np_data
        tensor = MyTensor(cudabuf, shape=np_data.shape, size=np_data.size)
        tensor.cpudata = np_data
        return tensor

    def to_host(self):
        if self.cpudata is None:
            raise Exception('not implemented')
        cuda.memcpy_dtoh(self.cpudata, self.gpudata)

def process(iH, iW, N, Ci, Co, kH=3, kW=3):
    np.random.seed(123)

    gpu_idx = 0

    platforms = cl.get_platforms()
    i = 0
    for platform in platforms:
       gpu_devices = platform.get_devices(device_type=cl.device_type.GPU)
       if gpu_idx < i + len(gpu_devices):
           ctx = cl.Context(devices=[gpu_devices[gpu_idx - i]])
           break
       i += len(gpu_devices)

    print('cl_context', ctx)
    q = cl.CommandQueue(ctx)

    W = np.random.randn(Ci,kH,kW,Co).astype(np.float32)
    W_cl = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=W)

    I = np.zeros((Ci,iH, iW,N), dtype=np.float32)
    I[:] = np.random.randn(*I.shape)
    I_cl = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=I)

    O = np.zeros((iH * iW * Co, N), dtype=np.float32)
    O_cl = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=O)

    gradO = np.random.randn(iH * iW * Co, N).astype(np.float32)
    gradO_cl = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=gradO)

    gradI = np.zeros((Ci, iH, iW, N), dtype=np.float32)
    gradI_cl = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=gradI)

    gradW = np.zeros((Ci, kH, kW, Co), dtype=np.float32)
    gradW_cl = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=gradW)

    convolver = api.Convolver(ctx, N, Ci, Co,
        kH, kW, iH, iW,
        kH // 2, kW // 2)
    convolver.fprop(ctx, q, I_cl, W_cl, O_cl)
    convolver.bprop_gradW(ctx, q,I_cl, gradO_cl, gradW_cl)
    # convolver.bprop_gradW(ctx, q,I_cl, gradO_cl, gradW_cl)

    cl.enqueue_copy(q, O, O_cl)
    cl.enqueue_copy(q, gradW, gradW_cl)
    cl.enqueue_copy(q, gradI, gradI_cl)

    #conv.deltas = gradI_cl
    #conv.dW = gradW_cl

    #print('gradO_cl', gradO_cl)
    #conv.bprop(gradO_cl)
#        cuda.Context.synchronize()

#    O = O_cl.get()
    #gradI_cl.to_host()
    #gradW_cl.to_host()
    return {
        'gradI': gradI, 'gradO': gradO, 'W': W,
        'gradW': gradW, 'O': O, 'I': I}

def simple1():
    image_size = 3
    N = 32
    Ci = 4
    Co = 4
    
    res = process(iH=image_size, iW=image_size, N=N, Ci=Ci,
        Co=Co)
    O = res['O']
    I = res['I']
    gradO = res['gradO']
    gradI = res['gradI']
    W = res['W']
    gradW = res['gradW']

    checkO(W=W, I=I, O=O, c=0, h=0, w=0, n=0)
    checkO(W=W, I=I, O=O, c=0, h=0, w=0, n=1)
    checkO(W=W, I=I, O=O, c=0, h=1, w=0, n=0)
    checkO(W=W, I=I, O=O, c=0, h=0, w=1, n=0)
    checkO(W=W, I=I, O=O, c=1, h=0, w=0, n=0)
    checkO(W=W, I=I, O=O, c=3, h=2, w=1, n=27)

    check_gradWeights(O=O, I=I, W=W, gradO=gradO, gradW=gradW, ci=0, h=0, w=0, co=0)
    check_gradWeights(O=O, I=I, W=W, gradO=gradO, gradW=gradW, ci=0, h=0, w=0, co=1)
    check_gradWeights(O=O, I=I, W=W, gradO=gradO, gradW=gradW, ci=0, h=0, w=1, co=0)
    check_gradWeights(O=O, I=I, W=W, gradO=gradO, gradW=gradW, ci=0, h=1, w=0, co=0)
    check_gradWeights(O=O, I=I, W=W, gradO=gradO, gradW=gradW, ci=0, h=0, w=0, co=0)

    check_gradI(O=O, I=I, W=W, gradO=gradO, gradI=gradI, c=0, h=0, w=0, n=0)
    check_gradI(O=O, I=I, W=W, gradO=gradO, gradI=gradI, c=0, h=0, w=0, n=1)
    check_gradI(O=O, I=I, W=W, gradO=gradO, gradI=gradI, c=0, h=0, w=1, n=0)
    check_gradI(O=O, I=I, W=W, gradO=gradO, gradI=gradI, c=0, h=1, w=0, n=0)
    check_gradI(O=O, I=I, W=W, gradO=gradO, gradI=gradI, c=0, h=0, w=0, n=0)

#        print('gradW', gradW)
def one():
    image_size = 64
    N = 128
    Ci = 32
    Co = 32

    res = process(iH=image_size, iW=image_size, N=N, Ci=Ci,
        Co=Co, kH=3, kW=3)
    O = res['O']
    I = res['I']
    gradO = res['gradO']
    gradI = res['gradI']
    W = res['W']
    gradW = res['gradW']

    checkO(W=W, I=I, O=O, c=0, h=0, w=0, n=0)
    checkO(W=W, I=I, O=O, c=0, h=0, w=0, n=1)
    checkO(W=W, I=I, O=O, c=0, h=1, w=0, n=0)
    checkO(W=W, I=I, O=O, c=0, h=0, w=1, n=0)
    checkO(W=W, I=I, O=O, c=1, h=0, w=0, n=0)
    checkO(W=W, I=I, O=O, c=3, h=2, w=1, n=27)

    check_gradWeights(O=O, I=I, W=W, gradO=gradO, gradW=gradW, ci=0, h=0, w=0, co=0)
    check_gradWeights(O=O, I=I, W=W, gradO=gradO, gradW=gradW, ci=0, h=0, w=0, co=1)
    check_gradWeights(O=O, I=I, W=W, gradO=gradO, gradW=gradW, ci=0, h=0, w=1, co=0)
    check_gradWeights(O=O, I=I, W=W, gradO=gradO, gradW=gradW, ci=0, h=1, w=0, co=0)
    check_gradWeights(O=O, I=I, W=W, gradO=gradO, gradW=gradW, ci=0, h=0, w=0, co=0)
    check_gradWeights(O=O, I=I, W=W, gradO=gradO, gradW=gradW, ci=19, h=2, w=1, co=17)

def two():
    image_size = 64
    N = 64
    Ci = 256
    Co = 256

    res = process(image_size=image_size, N=N, Ci=Ci,
        Co=Co)
    O = res['O']
    I = res['I']
    gradO = res['gradO']
    gradI = res['gradI']
    W = res['W']
    gradW = res['gradW']

    printDims(W=W, I=I)
    check_gradWeights(O=O, I=I, W=W, gradO=gradO, gradW=gradW, ci=0, h=0, w=0, co=0)
    check_gradWeights(O=O, I=I, W=W, gradO=gradO, gradW=gradW, ci=0, h=0, w=0, co=1)
    check_gradWeights(O=O, I=I, W=W, gradO=gradO, gradW=gradW, ci=0, h=0, w=1, co=0)
    check_gradWeights(O=O, I=I, W=W, gradO=gradO, gradW=gradW, ci=0, h=1, w=0, co=0)
    check_gradWeights(O=O, I=I, W=W, gradO=gradO, gradW=gradW, ci=0, h=0, w=0, co=0)
    check_gradWeights(O=O, I=I, W=W, gradO=gradO, gradW=gradW, ci=19, h=2, w=1, co=17)

def three():
    image_size = 32
    N = 32
    Ci = 512
    Co = 512

    res = process(image_size=image_size, N=N, Ci=Ci,
        Co=Co)
    O = res['O']
    I = res['I']
    gradO = res['gradO']
    gradI = res['gradI']
    W = res['W']
    gradW = res['gradW']

    printDims(W=W, I=I)
    check_gradWeights(O=O, I=I, W=W, gradO=gradO, gradW=gradW, ci=0, h=0, w=0, co=0)
    check_gradWeights(O=O, I=I, W=W, gradO=gradO, gradW=gradW, ci=0, h=0, w=0, co=1)
    check_gradWeights(O=O, I=I, W=W, gradO=gradO, gradW=gradW, ci=0, h=0, w=1, co=0)
    check_gradWeights(O=O, I=I, W=W, gradO=gradO, gradW=gradW, ci=0, h=1, w=0, co=0)
    check_gradWeights(O=O, I=I, W=W, gradO=gradO, gradW=gradW, ci=0, h=0, w=0, co=0)
    check_gradWeights(O=O, I=I, W=W, gradO=gradO, gradW=gradW, ci=19, h=2, w=1, co=17)

simple1()
#one()
#two()
#three()

