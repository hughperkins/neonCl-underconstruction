from winogradcl.layers.layer import Convolution
from winogradcl.backends.make_backend import make_backend
import numpy as np
from winogradcl.backends.kernels.cl.mycltensor import MyClTensor
import time

image_size = 64
batch_size = 128
input_filters = 32
output_filters = 32

np.random.seed(123)

with make_backend(batch_size=batch_size,
            datatype=np.float32, device_id=0) as be:
    conv = Convolution((3, 3, output_filters), strides=1, padding=1, be=be)
    print('created conv')
    W = np.random.randn(input_filters,3,3,output_filters).astype(np.float32)
    W_cl = MyClTensor.from_np(be, W)
    conv.W = W_cl

    print('type(W_cl)', type(W_cl))

    inputs = np.zeros((input_filters,image_size, image_size,batch_size), dtype=np.float32)
    inputs[:] = np.random.randn(*inputs.shape)
#    inputs_cl = gpuarray.to_gpu(inputs)
    inputs_cl = MyClTensor.from_np(be, inputs)

    print('type(inputs_cl)', type(inputs_cl))

    conv.configure((input_filters,image_size, image_size))
    print('configure done')
    outputs = np.zeros((image_size * image_size * output_filters, batch_size), dtype=np.float32)
    outputs_cl = MyClTensor.from_np(be, outputs)
    conv.outputs = outputs_cl
    conv.fprop(inputs_cl)
    for it in range(3):
      start = time.time()
      for i in range(10):
        conv.fprop(inputs_cl)
#      cuda.Context.synchronize()
      print('time=', time.time() - start)


    outputs_cl.to_host()
    print(outputs[1:3,1:3])
    print('outputs.shape', outputs.shape)

    assert abs(outputs[1,1] - 1.33960593) < 1e-4
    assert abs(outputs[1,2] + 6.06682396) < 1e-4
    assert abs(outputs[2,2] - 8.76905346) < 1e-4

