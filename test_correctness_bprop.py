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
import pycuda.driver as cuda
import pycuda.autoinit
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
        W_cuda = MyClTensor.from_np(be, W)

        inputs = np.zeros((input_filters,image_size, image_size,batch_size), dtype=np.float32)
        inputs[:] = np.random.randn(*inputs.shape)
        inputs_cuda = MyClTensor.from_np(be, inputs)

        outputs = np.zeros((image_size * image_size * output_filters, batch_size), dtype=np.float32)
        outputs_cuda = MyClTensor.from_np(be, outputs)

        conv = Convolution((3, 3, output_filters), strides=1, padding=1, be=be) #, init=init)
        print('created conv')

        conv.configure((input_filters,image_size, image_size))
        conv.W = W_cuda
        print('configure done')
        # we probably should do fprop first
        conv.outputs = outputs_cuda
        
        conv.fprop(inputs_cuda)
        outputs_cuda.to_host()

        gradOutputs = np.random.randn(image_size * image_size * output_filters, batch_size).astype(np.float32)
        gradOutputs_cuda = MyTensor.from_np(gradOutputs)

#        print('type(inputs_cuda)', type(inputs_cuda))

        gradInputs = np.zeros((input_filters,image_size, image_size,batch_size), dtype=np.float32)
#        gradInputs[:] = np.random.randn(*gradInputs.shape)
        gradInputs_cuda = MyTensor.from_np(gradInputs)

        gradW = np.zeros((input_filters,3,3,output_filters), dtype=np.float32)
        gradW_cuda = MyTensor.from_np(gradW)
        
        conv.deltas = gradInputs_cuda
        conv.dW = gradW_cuda

        print('gradOutputs_cuda', gradOutputs_cuda)
        conv.bprop(gradOutputs_cuda)
        cuda.Context.synchronize()

    #    outputs = outputs_cuda.get()
        gradInputs_cuda.to_host()
        gradW_cuda.to_host()
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
    
#        print('gradW', gradW)
    check_gradInputs(O=outputs, I=inputs, W=W, gradOutputs=gradOutputs, gradInputs=gradInputs, c=0, h=0, w=0, n=0)
    check_gradInputs(O=outputs, I=inputs, W=W, gradOutputs=gradOutputs, gradInputs=gradInputs, c=0, h=0, w=0, n=1)
    check_gradInputs(O=outputs, I=inputs, W=W, gradOutputs=gradOutputs, gradInputs=gradInputs, c=0, h=0, w=1, n=0)
    check_gradInputs(O=outputs, I=inputs, W=W, gradOutputs=gradOutputs, gradInputs=gradInputs, c=0, h=1, w=0, n=0)
    check_gradInputs(O=outputs, I=inputs, W=W, gradOutputs=gradOutputs, gradInputs=gradInputs, c=0, h=0, w=0, n=0)

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

    check_gradInputs(O=outputs, I=inputs, W=W, gradOutputs=gradOutputs, gradInputs=gradInputs, c=0, h=0, w=0, n=0)
    check_gradInputs(O=outputs, I=inputs, W=W, gradOutputs=gradOutputs, gradInputs=gradInputs, c=0, h=0, w=0, n=1)
    check_gradInputs(O=outputs, I=inputs, W=W, gradOutputs=gradOutputs, gradInputs=gradInputs, c=0, h=0, w=1, n=0)
    check_gradInputs(O=outputs, I=inputs, W=W, gradOutputs=gradOutputs, gradInputs=gradInputs, c=0, h=1, w=0, n=0)
    check_gradInputs(O=outputs, I=inputs, W=W, gradOutputs=gradOutputs, gradInputs=gradInputs, c=0, h=0, w=0, n=0)
    check_gradInputs(O=outputs, I=inputs, W=W, gradOutputs=gradOutputs, gradInputs=gradInputs, c=19, h=7, w=4, n=17)

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

    check(W=W, I=inputs, O=outputs, c=0, h=0, w=0, n=0)
    check(W=W, I=inputs, O=outputs, c=0, h=0, w=0, n=1)
    check(W=W, I=inputs, O=outputs, c=0, h=0, w=1, n=0)
    check(W=W, I=inputs, O=outputs, c=0, h=1, w=0, n=0)
    check(W=W, I=inputs, O=outputs, c=1, h=0, w=0, n=0)
    check(W=W, I=inputs, O=outputs, c=3, h=2, w=1, n=27)
    check(W=W, I=inputs, O=outputs, c=17, h=25, w=7, n=27)

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

    check(W=W, I=inputs, O=outputs, c=0, h=0, w=0, n=0, eps=1e-3)
    check(W=W, I=inputs, O=outputs, c=0, h=0, w=0, n=1, eps=1e-3)
    check(W=W, I=inputs, O=outputs, c=0, h=0, w=1, n=0, eps=1e-3)
    check(W=W, I=inputs, O=outputs, c=0, h=1, w=0, n=0, eps=1e-3)
    check(W=W, I=inputs, O=outputs, c=1, h=0, w=0, n=0, eps=1e-3)
    check(W=W, I=inputs, O=outputs, c=3, h=2, w=1, n=27, eps=1e-3)
    check(W=W, I=inputs, O=outputs, c=17, h=25, w=7, n=27, eps=1e-3)

simple1()
one()
#two()
#three()

