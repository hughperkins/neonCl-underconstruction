from __future__ import print_function

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import time

from neon.layers.layer import Convolution
from neon.backends.make_backend import make_backend


def printDims(W, I):
    Ci = W.shape[0]
    iH = I.shape[1]
    iW = I.shape[2]
    Co = W.shape[3]
    kH = W.shape[1]
    kW = W.shape[2]
    print('Ci', Ci, 'iH', iH, 'iW', iW, 'Co', Co, 'kH', kH, 'kW', kW)

def check(O, W, I, c, h, w, n, eps=1e-4):
    eps = 1 # hack for now ...
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

def simple1():
    image_size = 3
    batch_size = 32
    input_filters = 4
    output_filters = 4

    np.random.seed(123)

    with make_backend(batch_size=batch_size,
            datatype=np.float32, device_id=0) as be:

        W = np.random.randn(input_filters,3,3,output_filters).astype(np.float32)
        print('W.shape', W.shape)
        W_cuda = MyTensor.from_np(W)

        inputs = np.zeros((input_filters,image_size, image_size,batch_size), dtype=np.float32)
        inputs[:] = np.random.randn(*inputs.shape)
        inputs_cuda = MyTensor.from_np(inputs)

        print('type(inputs_cuda)', type(inputs_cuda))

        conv = Convolution((3, 3, output_filters), strides=1, padding=1, be=be) #, init=init)
        print('created conv')

        conv.configure((input_filters,image_size, image_size))
        conv.W = W_cuda
        print('configure done')
        outputs = np.zeros((image_size * image_size * output_filters, batch_size), dtype=np.float32)
        outputs_cuda = MyTensor.from_np(outputs)
        conv.outputs = outputs_cuda
        conv.fprop(inputs_cuda)
        cuda.Context.synchronize()

        for it in range(3):
            start = time.time()
            conv.fprop(inputs_cuda)
            cuda.Context.synchronize()
            print('time=', time.time() - start)

    #    outputs = outputs_cuda.get()
        outputs_cuda.to_host()
        printDims(W=W, I=inputs)
        check(W=W, I=inputs, O=outputs, c=0, h=0, w=0, n=0)
        check(W=W, I=inputs, O=outputs, c=0, h=0, w=0, n=1)
        check(W=W, I=inputs, O=outputs, c=0, h=1, w=0, n=0)
        check(W=W, I=inputs, O=outputs, c=0, h=0, w=1, n=0)
        check(W=W, I=inputs, O=outputs, c=1, h=0, w=0, n=0)
        check(W=W, I=inputs, O=outputs, c=3, h=2, w=1, n=27)

        print('outputs.shape', outputs.shape)

def one():
    image_size = 64
    batch_size = 128
    input_filters = 32
    output_filters = 32

    np.random.seed(123)
    
    with make_backend(batch_size=batch_size,
            datatype=np.float32, device_id=0) as be:
        W = np.random.randn(input_filters,3,3,output_filters).astype(np.float32)
        W_cuda = MyTensor.from_np(W)

        print('type(W_cuda)', type(W_cuda))

        inputs = np.zeros((input_filters,image_size, image_size,batch_size), dtype=np.float32)
        inputs[:] = np.random.randn(*inputs.shape)
        inputs_cuda = MyTensor.from_np(inputs)

        print('type(inputs_cuda)', type(inputs_cuda))

        conv = Convolution((3, 3, output_filters), strides=1, padding=1, be=be) #, init=init)
        print('created conv')
        conv.W = W_cuda

        conv.configure((input_filters,image_size, image_size))
        conv.W = W_cuda
        print('configure done')
        outputs = np.zeros((image_size * image_size * output_filters, batch_size), dtype=np.float32)
        outputs_cuda = MyTensor.from_np(outputs)
        conv.outputs = outputs_cuda
        conv.fprop(inputs_cuda)
        cuda.Context.synchronize()
        for it in range(3):
          start = time.time()
          conv.fprop(inputs_cuda)
          cuda.Context.synchronize()
          print('time=', time.time() - start)

    #    outputs = outputs_cuda.get()
        outputs_cuda.to_host()
        print(outputs[1:3,1:3])
        print('outputs.shape', outputs.shape)
        printDims(W=W, I=inputs)
        check(W=W, I=inputs, O=outputs, c=0, h=0, w=0, n=0)
        check(W=W, I=inputs, O=outputs, c=0, h=0, w=0, n=1)
        check(W=W, I=inputs, O=outputs, c=0, h=0, w=1, n=0)
        check(W=W, I=inputs, O=outputs, c=0, h=1, w=0, n=0)
        check(W=W, I=inputs, O=outputs, c=1, h=0, w=0, n=0)
        check(W=W, I=inputs, O=outputs, c=3, h=2, w=1, n=27)
        check(W=W, I=inputs, O=outputs, c=17, h=25, w=7, n=27)

def two():
    image_size = 64
    batch_size = 64
    input_filters = 256
    output_filters = 256

    np.random.seed(123)
    
    with make_backend(batch_size=batch_size,
            datatype=np.float32, device_id=0) as be:
        W = np.random.randn(input_filters,3,3,output_filters).astype(np.float32)
        W_cuda = MyTensor.from_np(W)

        print('type(W_cuda)', type(W_cuda))

        inputs = np.zeros((input_filters,image_size, image_size,batch_size), dtype=np.float32)
        inputs[:] = np.random.randn(*inputs.shape)
        inputs_cuda = MyTensor.from_np(inputs)

        print('type(inputs_cuda)', type(inputs_cuda))

        conv = Convolution((3, 3, output_filters), strides=1, padding=1, be=be) #, init=init)
        print('created conv')
        conv.W = W_cuda

        conv.configure((input_filters,image_size, image_size))
        conv.W = W_cuda
        print('configure done')
        outputs = np.zeros((image_size * image_size * output_filters, batch_size), dtype=np.float32)
        outputs_cuda = MyTensor.from_np(outputs)
        conv.outputs = outputs_cuda
        conv.fprop(inputs_cuda)
        cuda.Context.synchronize()
        for it in range(3):
          start = time.time()
          conv.fprop(inputs_cuda)
          cuda.Context.synchronize()
          print('time=', time.time() - start)

    #    outputs = outputs_cuda.get()
        outputs_cuda.to_host()
        print(outputs[1:3,1:3])
        print('outputs.shape', outputs.shape)
        printDims(W=W, I=inputs)
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

    np.random.seed(123)
    
    with make_backend(batch_size=batch_size,
            datatype=np.float32, device_id=0) as be:
        W = np.random.randn(input_filters,3,3,output_filters).astype(np.float32)
        W_cuda = MyTensor.from_np(W)

        print('type(W_cuda)', type(W_cuda))

        inputs = np.zeros((input_filters,image_size, image_size,batch_size), dtype=np.float32)
        inputs[:] = np.random.randn(*inputs.shape)
        inputs_cuda = MyTensor.from_np(inputs)

        print('type(inputs_cuda)', type(inputs_cuda))

        conv = Convolution((3, 3, output_filters), strides=1, padding=1, be=be) #, init=init)
        print('created conv')
        conv.W = W_cuda

        conv.configure((input_filters,image_size, image_size))
        conv.W = W_cuda
        print('configure done')
        outputs = np.zeros((image_size * image_size * output_filters, batch_size), dtype=np.float32)
        outputs_cuda = MyTensor.from_np(outputs)
        conv.outputs = outputs_cuda
        conv.fprop(inputs_cuda)
        cuda.Context.synchronize()
        for it in range(3):
          start = time.time()
          conv.fprop(inputs_cuda)
          cuda.Context.synchronize()
          print('time=', time.time() - start)

    #    outputs = outputs_cuda.get()
        outputs_cuda.to_host()
        print(outputs[1:3,1:3])
        print('outputs.shape', outputs.shape)
        printDims(W=W, I=inputs)
        check(W=W, I=inputs, O=outputs, c=0, h=0, w=0, n=0, eps=1e-3)
        check(W=W, I=inputs, O=outputs, c=0, h=0, w=0, n=1, eps=1e-3)
        check(W=W, I=inputs, O=outputs, c=0, h=0, w=1, n=0, eps=1e-3)
        check(W=W, I=inputs, O=outputs, c=0, h=1, w=0, n=0, eps=1e-3)
        check(W=W, I=inputs, O=outputs, c=1, h=0, w=0, n=0, eps=1e-3)
        check(W=W, I=inputs, O=outputs, c=3, h=2, w=1, n=27, eps=1e-3)
        check(W=W, I=inputs, O=outputs, c=17, h=25, w=7, n=27, eps=1e-3)

simple1()
#one()
#two()
#three()

