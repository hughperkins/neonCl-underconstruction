from neon.layers import Convolution
#from neon.initializers import Gaussian
from neon.backends import gen_backend
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import time




#init = Gaussian()

def calc(O, W, I, c, h, w, n):
    Ci = W.shape[0]
    iH = I.shape[1]
    iW = I.shape[2]
    Co = W.shape[3]
    kH = W.shape[1]
    kW = W.shape[2]
    print('Ci', Ci, 'iH', iH, 'iW', iW, 'Co', Co, 'kH', kH, 'kW', kW)
    print('c', c, 'h', h, 'w', w, 'n', n)
    
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
    print('cpu %.6f gpu %.6f' % (sum, gpu_value))
    assert abs(sum - gpu_value) < 1e-4
    return ""

def simple1():
    image_size = 3
    batch_size = 32
    input_filters = 4
    output_filters = 4

    np.random.seed(123)

    gen_backend(batch_size=batch_size,
            datatype=np.float32, device_id=0)

    W = np.random.randn(input_filters,3,3,output_filters).astype(np.float32)
    W_cuda = gpuarray.to_gpu(W)

    print('type(W_cuda)', type(W_cuda))

    #conv.W = np.random.randn(input_filters,3,3,output_filters).astype(np.float32)

    #inputs = np.zeros((batch_size,image_size, image_size,input_filters), dtype=np.float32)
    inputs = np.zeros((input_filters,image_size, image_size,batch_size), dtype=np.float32)
    inputs[:] = np.random.randn(*inputs.shape)
    inputs_cuda = gpuarray.to_gpu(inputs)

    print('type(inputs_cuda)', type(inputs_cuda))

    conv = Convolution((3, 3, output_filters), strides=1, padding=1) #, init=init)
    print('created conv')
    conv.W = W_cuda

    conv.configure((input_filters,image_size, image_size))
    conv.W = W_cuda
    print('configure done')
    #conv.allocate()
    #print('conv.outputs.shape', conv.outputs.shape)
    #print('type(conv.outputs)', type(conv.outputs))
    #print('type(conv.outputs.gpudata)', type(conv.outputs.gpudata))
    #print('allocate done')
    outputs = np.zeros((image_size * image_size * output_filters, batch_size), dtype=np.float32)
    outputs_cuda = gpuarray.to_gpu(outputs)
    conv.outputs = outputs_cuda
    conv.fprop(inputs_cuda)
    
    for it in range(3):
        start = time.time()
        conv.fprop(inputs_cuda)
        cuda.Context.synchronize()
        print('time=', time.time() - start)

    outputs = outputs_cuda.get()
    print(outputs)
    print(outputs[:,0])

    print(calc(W=W, I=inputs, O=outputs, c=0, h=0, w=0, n=0))
    print(calc(W=W, I=inputs, O=outputs, c=0, h=0, w=0, n=1))
    print(calc(W=W, I=inputs, O=outputs, c=0, h=1, w=0, n=0))
    print(calc(W=W, I=inputs, O=outputs, c=0, h=0, w=1, n=0))
    print(calc(W=W, I=inputs, O=outputs, c=1, h=0, w=0, n=0))
    print(calc(W=W, I=inputs, O=outputs, c=3, h=2, w=1, n=27))

    print('outputs.shape', outputs.shape)

def one():
    image_size = 64
    batch_size = 128
    input_filters = 32
    output_filters = 32

    np.random.seed(123)
    
    gen_backend(batch_size=batch_size,
            datatype=np.float32, device_id=0)

    W = np.random.randn(input_filters,3,3,output_filters).astype(np.float32)
    W_cuda = gpuarray.to_gpu(W)

    print('type(W_cuda)', type(W_cuda))

    #conv.W = np.random.randn(input_filters,3,3,output_filters).astype(np.float32)

    #inputs = np.zeros((batch_size,image_size, image_size,input_filters), dtype=np.float32)
    inputs = np.zeros((input_filters,image_size, image_size,batch_size), dtype=np.float32)
    inputs[:] = np.random.randn(*inputs.shape)
    inputs_cuda = gpuarray.to_gpu(inputs)

    print('type(inputs_cuda)', type(inputs_cuda))

    conv = Convolution((3, 3, output_filters), strides=1, padding=1) #, init=init)
    print('created conv')
    conv.W = W_cuda

    conv.configure((input_filters,image_size, image_size))
    conv.W = W_cuda
    print('configure done')
    #conv.allocate()
    #print('conv.outputs.shape', conv.outputs.shape)
    #print('type(conv.outputs)', type(conv.outputs))
    #print('type(conv.outputs.gpudata)', type(conv.outputs.gpudata))
    #print('allocate done')
    outputs = np.zeros((image_size * image_size * output_filters, batch_size), dtype=np.float32)
    outputs_cuda = gpuarray.to_gpu(outputs)
    conv.outputs = outputs_cuda
    conv.fprop(inputs_cuda)
    for it in range(3):
      start = time.time()
      conv.fprop(inputs_cuda)
      cuda.Context.synchronize()
      print('time=', time.time() - start)


#    outputs = outputs_cuda.get()

#    assert abs(outputs[1,1] - 1.33960593) < 1e-4
#    assert abs(outputs[1,2] + 6.06682396) < 1e-4
#    assert abs(outputs[2,2] - 8.76905346) < 1e-4

    outputs = outputs_cuda.get()
    print(outputs[1:3,1:3])
    print('outputs.shape', outputs.shape)
    print(calc(W=W, I=inputs, O=outputs, c=0, h=0, w=0, n=0))
    print(calc(W=W, I=inputs, O=outputs, c=0, h=0, w=0, n=1))
    print(calc(W=W, I=inputs, O=outputs, c=0, h=0, w=1, n=0))
    print(calc(W=W, I=inputs, O=outputs, c=0, h=1, w=0, n=0))
    print(calc(W=W, I=inputs, O=outputs, c=1, h=0, w=0, n=0))
    print(calc(W=W, I=inputs, O=outputs, c=3, h=2, w=1, n=27))
    print(calc(W=W, I=inputs, O=outputs, c=17, h=25, w=7, n=27))

simple1()
one()

