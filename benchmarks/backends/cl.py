# tests using opencl backend, in winogradcl branch

import sys
import time
import importlib
import numpy as np
from mycltensor import MyClTensor
from neon.layers.layer import Convolution
from neon.backends.make_backend import make_backend

def test(batch_size, its, layer):
    np.random.seed(123)

    input_filters = layer['Ci']
    output_filters = layer['Co']
    image_size = layer['iW']
    assert layer['iH'] == image_size
    assert layer['kH'] == layer['kW'] == 3
    with make_backend(batch_size=batch_size,
            datatype=np.float32, device_id=0) as be:
        W = np.random.randn(input_filters,3,3,output_filters).astype(np.float32)
        W_cl = MyClTensor.from_np(be, W)

        inputs = np.zeros((input_filters,image_size, image_size, batch_size), dtype=np.float32)
        inputs[:] = np.random.randn(*inputs.shape)
        inputs_cl = MyClTensor.from_np(be, inputs)

        conv = Convolution((3, 3, output_filters), strides=1, padding=1, be=be) #, init=init)

        conv.configure((input_filters,image_size, image_size))
        conv.W = W_cl

        outputs = np.zeros((image_size * image_size * output_filters, batch_size), dtype=np.float32)
        outputs_cl = MyClTensor.from_np(be, outputs)
        conv.outputs = outputs_cl
        conv.fprop(inputs_cl)
        be.q.finish()

        gradOutputs = np.random.randn(image_size * image_size * output_filters, batch_size).astype(np.float32)
        gradOutputs_cl = MyClTensor.from_np(be, gradOutputs)

        gradInputs = np.zeros((input_filters,image_size, image_size,batch_size), dtype=np.float32)
        gradInputs_cl = MyClTensor.from_np(be, gradInputs)

        gradW = np.zeros((input_filters,3,3,output_filters), dtype=np.float32)
        gradW_cl = MyClTensor.from_np(be, gradW)
        
        conv.deltas = gradInputs_cl
        conv.dW = gradW_cl

        conv.bprop(gradOutputs_cl)

        fpropCumTime = 0
        bpropCumTime = 0
        for it in range(its):
            start = time.time()
            conv.fprop(inputs_cl)
            be.q.finish()
            end = time.time()
            fprop = end - start
            fpropCumTime += end - start
            
            start = time.time()
            conv.bprop(gradOutputs_cl)
            end = time.time()
            bprop = end - start
            bpropCumTime += end - start
            print('fprop %.3f bprop %.3f' % (fprop, bprop))
        print('avg fprop %.3f bprop %.3f' % (fpropCumTime / its, bpropCumTime / its))

