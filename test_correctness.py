from __future__ import print_function

import numpy as np
import time

from winogradcl.backends.kernels.cl.mycltensor import MyClTensor
from winogradcl.layers.layer import Convolution
from winogradcl.backends.make_backend import make_backend

its = 1

def printDims(W, I):
    Ci = W.shape[0]
    iH = I.shape[1]
    iW = I.shape[2]
    Co = W.shape[3]
    kH = W.shape[1]
    kW = W.shape[2]
    print('Ci', Ci, 'iH', iH, 'iW', iW, 'Co', Co, 'kH', kH, 'kW', kW)

def check(O, W, I, c, h, w, n, eps=1e-4):
#    eps = 1e6 # hack for now ...
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

        print('type(inputs_cl)', type(inputs_cl))

        conv = Convolution((3, 3, output_filters), strides=1, padding=1, be=be) #, init=init)
        print('created conv')

        conv.configure((input_filters,image_size, image_size))
        conv.W = W_cl
        print('configure done')
        outputs = np.zeros((image_size * image_size * output_filters, batch_size), dtype=np.float32)
        outputs_cl = MyClTensor.from_np(be, outputs)
        conv.outputs = outputs_cl
        conv.fprop(inputs_cl)
#        cuda.Context.synchronize()

        for it in range(its):
            start = time.time()
            conv.fprop(inputs_cl)
#            cuda.Context.synchronize()
            print('time=', time.time() - start)

        outputs_cl.to_host()
        return {
            'W': W,
            'outputs': outputs, 'inputs': inputs}

def simple1():
    image_size = 3
    batch_size = 32
    input_filters = 4
    output_filters = 4

    res = process(image_size=image_size, batch_size=batch_size, input_filters=input_filters,
        output_filters=output_filters)
    outputs = res['outputs']
    inputs = res['inputs']
    W = res['W']

    #    outputs = outputs_cl.get()
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

    res = process(image_size=image_size, batch_size=batch_size, input_filters=input_filters,
        output_filters=output_filters)
    outputs = res['outputs']
    inputs = res['inputs']
    W = res['W']

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

    res = process(image_size=image_size, batch_size=batch_size, input_filters=input_filters,
        output_filters=output_filters)
    outputs = res['outputs']
    inputs = res['inputs']
    W = res['W']

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

    res = process(image_size=image_size, batch_size=batch_size, input_filters=input_filters,
        output_filters=output_filters)
    outputs = res['outputs']
    inputs = res['inputs']
    W = res['W']

    check(W=W, I=inputs, O=outputs, c=0, h=0, w=0, n=0, eps=1e-3)
    check(W=W, I=inputs, O=outputs, c=0, h=0, w=0, n=1, eps=1e-3)
    check(W=W, I=inputs, O=outputs, c=0, h=0, w=1, n=0, eps=1e-3)
    check(W=W, I=inputs, O=outputs, c=0, h=1, w=0, n=0, eps=1e-3)
    check(W=W, I=inputs, O=outputs, c=1, h=0, w=0, n=0, eps=1e-3)
    check(W=W, I=inputs, O=outputs, c=3, h=2, w=1, n=27, eps=1e-3)
    check(W=W, I=inputs, O=outputs, c=17, h=25, w=7, n=27, eps=1e-3)

def four():
    image_size = 224
    batch_size = 128
    input_filters = 32
    output_filters = 32

    res = process(image_size=image_size, batch_size=batch_size, input_filters=input_filters,
        output_filters=output_filters)
    outputs = res['outputs']
    inputs = res['inputs']
    W = res['W']

    check(W=W, I=inputs, O=outputs, c=0, h=0, w=0, n=0, eps=1e-3)
    check(W=W, I=inputs, O=outputs, c=0, h=0, w=0, n=1, eps=1e-3)
    check(W=W, I=inputs, O=outputs, c=0, h=0, w=1, n=0, eps=1e-3)
    check(W=W, I=inputs, O=outputs, c=0, h=1, w=0, n=0, eps=1e-3)
    check(W=W, I=inputs, O=outputs, c=1, h=0, w=0, n=0, eps=1e-3)
    check(W=W, I=inputs, O=outputs, c=3, h=2, w=1, n=27, eps=1e-3)
    check(W=W, I=inputs, O=outputs, c=17, h=25, w=7, n=27, eps=1e-3)

#simple1()
#one()
two()
#three()
#four()

