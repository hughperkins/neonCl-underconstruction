import numpy as np
import random

def ceil_div(x, y):
    return -(-x // y)

A = np.random.randn(3,7,4,5).astype(np.float32)
# print('a', a)

a2 = np.copy(A)
print(A[2,2,1,4])
a2[2,2,1,4] = 123
print(A[2,2,1,4])
print(a2[2,2,1,4])

ar = a2.reshape(3*7*4*5)
print('ar.shape', ar.shape)

arc = np.zeros(*ar.shape, dtype=np.float32)
print('arc.shape', arc.shape)

for a in range(3):
    for b in range(28):
        for c in range(5):
            arc[c * 28 * 3 + b * 3 + a] = ar[a * 28 * 5 + b * 5 + c]

arc = arc.reshape(5,7,4,3)

for it in range(5):
  a = random.randint(0, 3-1)
  b = random.randint(0, 7-1)
  c = random.randint(0, 4-1)
  d = random.randint(0, 5-1)
  c1 = A[a,b,c,d]
  c2 = arc[d,b,c,a]
  print(c1, c2, c1-c2)
  assert c1 == c2

import pyopencl as cl

gpu_idx = 0

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

a3 = np.copy(A)
a3_gpu = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a3)

at = np.zeros((5,7,4,3), dtype=np.float32)
at_gpu = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=at)

from winogradcl.backends.kernels.cl.clshuffler import get_shuffle_kernel_d3_cl
kernel = get_shuffle_kernel_d3_cl(ctx, 'f4')

from winogradcl.backends.kernels.cl.callkernel import call_cl_kernel

# (3,7,4,5)
call_cl_kernel(kernel, q,
    (ceil_div(5, 32), ceil_div(3, 32), 7 * 4),
    (32, 8, 1),
    at_gpu, a3_gpu,
    7*4*5, 5, 3*7*4, 3)

cl.enqueue_copy(q, at, at_gpu)
at = at.reshape(5,7,4,3)

for it in range(5):
  a = random.randint(0, 3-1)
  b = random.randint(0, 7-1)
  c = random.randint(0, 4-1)
  d = random.randint(0, 5-1)
  c1 = A[a,b,c,d]
  c2 = at[d,b,c,a]
  print(c1, c2, c1-c2)
  assert c1 == c2

a4 = np.copy(A)
a4_gpu = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a4)

a4t = np.zeros((5,7,4,3), dtype=np.float32)
a4t_gpu = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a4t)

from winogradcl import api
shuffler = api.Shuffler(ctx, (3, 7, 4, 5))
# api.shuffle(ctx, q, a4_gpu, (3, 7, 4, 5), a4t_gpu)
shuffler.shuffle(q, a4t_gpu, a4_gpu)
cl.enqueue_copy(q, a4t, a4t_gpu)

for it in range(5):
  a = random.randint(0, 3-1)
  b = random.randint(0, 7-1)
  c = random.randint(0, 4-1)
  d = random.randint(0, 5-1)
  c1 = A[a,b,c,d]
  c2 = a4t[d,b,c,a]
  print(c1, c2, c1-c2)
  assert c1 == c2

