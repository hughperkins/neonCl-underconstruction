import winogradcl.api
import numpy as np
import time
import pyopencl as cl

image_size = 64
batch_size = 128
input_filters = 32
output_filters = 32

gpu_idx = 0

np.random.seed(123)

platforms = cl.get_platforms()
i = 0
for platform in platforms:
   gpu_devices = platform.get_devices(device_type=cl.device_type.GPU)
   if gpu_idx < i + len(gpu_devices):
       ctx = cl.Context(devices=[gpu_devices[gpu_idx-i]])
       break
   i += len(gpu_devices)

print('context', ctx)
#ctx = cl.create_some_context()
q = cl.CommandQueue(ctx)

mf = cl.mem_flags

W = np.random.randn(input_filters,3,3,output_filters).astype(np.float32)
W_cl = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=W)

# create as NHWC
inputs = np.zeros((batch_size,image_size, image_size,input_filters), dtype=np.float32)
inputs[:] = np.random.randn(*inputs.shape)
inputs_cl = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=inputs)

outputs = np.zeros((image_size * image_size * output_filters, batch_size), dtype=np.float32)
outputs_cl = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=outputs)

winogradcl.api.fprop(ctx, q, I=inputs_cl, I_layout='N H W C', W=W_cl, W_layout='Ci H W Co', O=outputs_cl, O_layout='N H W C')

cl.enqueue_copy(q, outputs, outputs_cl)

print(outputs[1:3,1:3])
print('outputs.shape', outputs.shape)

