from neon.layers.layer import Convolution
from neon.backends import Backend
import numpy as np
import pycuda.driver as cuda
#import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import time

image_size = 64
batch_size = 128
input_filters = 32
output_filters = 32

np.random.seed(123)

with Backend(batch_size=batch_size,
            datatype=np.float32, device_id=0) as be:
    conv = Convolution((3, 3, output_filters), strides=1, padding=1, be=be)
    print('created conv')
    W = np.random.randn(input_filters,3,3,output_filters).astype(np.float32)
    W_cuda = gpuarray.to_gpu(W)
    conv.W = W_cuda

    print('type(W_cuda)', type(W_cuda))

    inputs = np.zeros((input_filters,image_size, image_size,batch_size), dtype=np.float32)
    inputs[:] = np.random.randn(*inputs.shape)
    inputs_cuda = gpuarray.to_gpu(inputs)

    print('type(inputs_cuda)', type(inputs_cuda))

    conv.configure((input_filters,image_size, image_size))
    print('configure done')
    outputs = np.zeros((image_size * image_size * output_filters, batch_size), dtype=np.float32)
    outputs_cuda = gpuarray.to_gpu(outputs)
    conv.outputs = outputs_cuda
    conv.fprop(inputs_cuda)
    for it in range(3):
      start = time.time()
      for i in range(10):
        conv.fprop(inputs_cuda)
      cuda.Context.synchronize()
      print('time=', time.time() - start)


    outputs = outputs_cuda.get()
    print(outputs[1:3,1:3])

    assert abs(outputs[1,1] - 1.33960593) < 1e-4
    assert abs(outputs[1,2] + 6.06682396) < 1e-4
    assert abs(outputs[2,2] - 8.76905346) < 1e-4

