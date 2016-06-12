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
#import pycuda.driver as cuda
#import pycuda.autoinit
import time

from mycltensor import MyClTensor
from neon.layers.layer import Convolution
from neon.backends.make_backend import make_backend

its = 1

def printDims(W, I):
    Ci = W.shape[0]
    iH = I.shape[1]
    iW = I.shape[2]
    Co = W.shape[3]
    kH = W.shape[1]
    kW = W.shape[2]
    print('Ci', Ci, 'iH', iH, 'iW', iW, 'Co', Co, 'kH', kH, 'kW', kW)

def check_gradWeights(O, I, W, gradOutputs, gradW, ci, h, w, co, eps=1e-2):
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
                    v = I[ci, ih, iw, n] * gradOutputs[co * iH * iW + oh * iW + ow, n]
                    sum += v
    cpu_value = sum
    gpu_value = gradW[ci, kh, kw, co]
    print('gpu', gpu_value, 'cpu', cpu_value)
    assert abs(cpu_value - gpu_value) < eps

def check_gradInputs(O, I, W, gradOutputs, gradInputs, c, h, w, n, eps=1e-4):
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
                    v = gradOutputs[co * iH * iW + oh * iW + ow, n] * W[ci, kh, kw, co]
                    sum += v
    cpu_value = sum
    gpu_value = gradInputs[c, ih, iw, n]
    print('gpu', gpu_value, 'cpu', cpu_value)
    assert abs(cpu_value - gpu_value) < eps

def check(O, W, I, c, h, w, n, eps=1e-4):
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
    gpu_value = O[c*iH*iW + h*iW + w,n]
    print('c', c, 'h', h, 'w', w, 'n', n, 'cpu %.6f gpu %.6f' % (sum, gpu_value))
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

def process(image_size, batch_size, input_filters, output_filters):
    np.random.seed(123)

    with make_backend(batch_size=batch_size,
            datatype=np.float32, device_id=0) as be:

        W = np.random.randn(input_filters,3,3,output_filters).astype(np.float32)
        print('W.shape', W.shape)
        W_cl = MyClTensor.from_np(be, W)

        inputs = np.zeros((input_filters,image_size, image_size,batch_size), dtype=np.float32)
        inputs[:] = np.random.randn(*inputs.shape)
        inputs_cl = MyClTensor.from_np(be, inputs)

        outputs = np.zeros((image_size * image_size * output_filters, batch_size), dtype=np.float32)
        outputs_cl = MyClTensor.from_np(be, outputs)

        conv = Convolution((3, 3, output_filters), strides=1, padding=1, be=be) #, init=init)
        print('created conv')

        conv.configure((input_filters,image_size, image_size))
        conv.W = W_cl
        print('configure done')
        # we probably should do fprop first
        conv.outputs = outputs_cl
        
        conv.fprop(inputs_cl)
        outputs_cl.to_host()

        gradOutputs = np.random.randn(image_size * image_size * output_filters, batch_size).astype(np.float32)
        gradOutputs_cl = MyClTensor.from_np(be, gradOutputs)

#        print('type(inputs_cl)', type(inputs_cl))

        gradInputs = np.zeros((input_filters,image_size, image_size,batch_size), dtype=np.float32)
#        gradInputs[:] = np.random.randn(*gradInputs.shape)
        gradInputs_cl = MyClTensor.from_np(be, gradInputs)

        gradW = np.zeros((input_filters,3,3,output_filters), dtype=np.float32)
        gradW_cl = MyClTensor.from_np(be, gradW)
        
        conv.deltas = gradInputs_cl
        conv.dW = gradW_cl

        print('gradOutputs_cl', gradOutputs_cl)
        conv.bprop(gradOutputs_cl)
#        cuda.Context.synchronize()

    #    outputs = outputs_cl.get()
        gradInputs_cl.to_host()
        gradW_cl.to_host()
        return {
            'gradInputs': gradInputs, 'gradOutputs': gradOutputs, 'W': W,
            'gradW': gradW, 'outputs': outputs, 'inputs': inputs}

def simple1():
    image_size = 3
    batch_size = 32
    input_filters = 4
    output_filters = 4
    
    res = process(image_size=image_size, batch_size=batch_size, input_filters=input_filters,
        output_filters=output_filters)
    outputs = res['outputs']
    inputs = res['inputs']
    gradOutputs = res['gradOutputs']
    gradInputs = res['gradInputs']
    W = res['W']
    gradW = res['gradW']
    
#        print('gradW', gradW)
    check_gradWeights(O=outputs, I=inputs, W=W, gradOutputs=gradOutputs, gradW=gradW, ci=0, h=0, w=0, co=0)
    check_gradWeights(O=outputs, I=inputs, W=W, gradOutputs=gradOutputs, gradW=gradW, ci=0, h=0, w=0, co=1)
    check_gradWeights(O=outputs, I=inputs, W=W, gradOutputs=gradOutputs, gradW=gradW, ci=0, h=0, w=1, co=0)
    check_gradWeights(O=outputs, I=inputs, W=W, gradOutputs=gradOutputs, gradW=gradW, ci=0, h=1, w=0, co=0)
    check_gradWeights(O=outputs, I=inputs, W=W, gradOutputs=gradOutputs, gradW=gradW, ci=0, h=0, w=0, co=0)

def one():
    image_size = 64
    batch_size = 128
    input_filters = 32
    output_filters = 32

    res = process(image_size=image_size, batch_size=batch_size, input_filters=input_filters,
        output_filters=output_filters)
    outputs = res['outputs']
    inputs = res['inputs']
    gradOutputs = res['gradOutputs']
    gradInputs = res['gradInputs']
    W = res['W']
    gradW = res['gradW']

    check_gradWeights(O=outputs, I=inputs, W=W, gradOutputs=gradOutputs, gradW=gradW, ci=0, h=0, w=0, co=0)
    check_gradWeights(O=outputs, I=inputs, W=W, gradOutputs=gradOutputs, gradW=gradW, ci=0, h=0, w=0, co=1)
    check_gradWeights(O=outputs, I=inputs, W=W, gradOutputs=gradOutputs, gradW=gradW, ci=0, h=0, w=1, co=0)
    check_gradWeights(O=outputs, I=inputs, W=W, gradOutputs=gradOutputs, gradW=gradW, ci=0, h=1, w=0, co=0)
    check_gradWeights(O=outputs, I=inputs, W=W, gradOutputs=gradOutputs, gradW=gradW, ci=0, h=0, w=0, co=0)
    check_gradWeights(O=outputs, I=inputs, W=W, gradOutputs=gradOutputs, gradW=gradW, ci=19, h=2, w=1, co=17)

def two():
    image_size = 64
    batch_size = 64
    input_filters = 256
    output_filters = 256

    res = process(image_size=image_size, batch_size=batch_size, input_filters=input_filters,
        output_filters=output_filters)
    outputs = res['outputs']
    inputs = res['inputs']
    gradOutputs = res['gradOutputs']
    gradInputs = res['gradInputs']
    W = res['W']
    gradW = res['gradW']

    printDims(W=W, I=inputs)
    check_gradWeights(O=outputs, I=inputs, W=W, gradOutputs=gradOutputs, gradW=gradW, ci=0, h=0, w=0, co=0)
    check_gradWeights(O=outputs, I=inputs, W=W, gradOutputs=gradOutputs, gradW=gradW, ci=0, h=0, w=0, co=1)
    check_gradWeights(O=outputs, I=inputs, W=W, gradOutputs=gradOutputs, gradW=gradW, ci=0, h=0, w=1, co=0)
    check_gradWeights(O=outputs, I=inputs, W=W, gradOutputs=gradOutputs, gradW=gradW, ci=0, h=1, w=0, co=0)
    check_gradWeights(O=outputs, I=inputs, W=W, gradOutputs=gradOutputs, gradW=gradW, ci=0, h=0, w=0, co=0)
    check_gradWeights(O=outputs, I=inputs, W=W, gradOutputs=gradOutputs, gradW=gradW, ci=19, h=2, w=1, co=17)

def three():
    image_size = 32
    batch_size = 32
    input_filters = 512
    output_filters = 512

    res = process(image_size=image_size, batch_size=batch_size, input_filters=input_filters,
        output_filters=output_filters)
    outputs = res['outputs']
    inputs = res['inputs']
    gradOutputs = res['gradOutputs']
    gradInputs = res['gradInputs']
    W = res['W']
    gradW = res['gradW']

    printDims(W=W, I=inputs)
    check_gradWeights(O=outputs, I=inputs, W=W, gradOutputs=gradOutputs, gradW=gradW, ci=0, h=0, w=0, co=0)
    check_gradWeights(O=outputs, I=inputs, W=W, gradOutputs=gradOutputs, gradW=gradW, ci=0, h=0, w=0, co=1)
    check_gradWeights(O=outputs, I=inputs, W=W, gradOutputs=gradOutputs, gradW=gradW, ci=0, h=0, w=1, co=0)
    check_gradWeights(O=outputs, I=inputs, W=W, gradOutputs=gradOutputs, gradW=gradW, ci=0, h=1, w=0, co=0)
    check_gradWeights(O=outputs, I=inputs, W=W, gradOutputs=gradOutputs, gradW=gradW, ci=0, h=0, w=0, co=0)
    check_gradWeights(O=outputs, I=inputs, W=W, gradOutputs=gradOutputs, gradW=gradW, ci=19, h=2, w=1, co=17)

simple1()
one()
#two()
#three()
