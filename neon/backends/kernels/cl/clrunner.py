"""
this is going to receive a cuda buffer
copy it to an np buffer
copy it to a cl buffer
call the cl fprop
then reverse the process

It's going to be mega-fast :-P

but it'll at least show the cl fprop is/isnt working
"""

import time
import numpy as np
import pyopencl as cl
import pycuda.driver as cuda
from neon.backends.kernels.cl import convolution_cl

gpu_idx = 0  # hardcode this for now...

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

class ClRunner(object):
    def __init__(self, dtype, filter_size, bsum, operation):
        self.dtype = dtype
        self.filter_size = filter_size
        self.bsum = bsum
        self.operation = operation
        self.kernel = convolution_cl._get_conv_kernel(
            ctx=ctx, options='', dtype=self.dtype, filter_size=self.filter_size,
            bsum=self.bsum, operation=self.operation)

    def execute(self, grid, block, stream, alpha, beta, Igpudata, Fgpudata, Ogpudata, bsum_gpudata,
        C, D, H, W, N, T, R, S, K, M, P, Q,
        str_w, str_h, pad_w, pad_h, HWN, KRST, PQN,
        PQ, zeroa, zerob, magic_PQ, shift_PQ, magic_Q, shift_Q, magic_S, shift_S,
        *args, shared_size):
        print('grid', grid, 'block', block, 'stream', stream, 'alpha', alpha, 'beta', beta,
              'Igpudata', Igpudata, 'Fgpudata', Fgpudata,
              'Ogpudata', Ogpudata, 'bsum_gpudata', bsum_gpudata)
        print('C', C, 'D', D, 'H', H, 'W', W, 'N', N, 'T', T, 'R', R, 'S', S, 'K', K, 'M', M, 'P', P, 'Q', Q)
        print('str_w', str_w, 'str_h', str_h, 'pad_w', pad_w, 'pad_h', pad_h, 'HWN', HWN, 'KRST', KRST, 'PQN', PQN)
        print('PQ', PQ, 'zeroa', zeroa, 'zerob', zerob)
        print('magic_PQ', magic_PQ, 'shift_PQ', shift_PQ, 'magic_Q', magic_Q, 'shift_Q', shift_Q)
        print('magic_S', magic_S, 'shift_S', shift_S)
        print('args', *args, 'shared_size', shared_size)
#        print('cuda_buffer', cuda_buffer)
#         cpu_buffer = np.zeros(
#        cpu_buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a)
        I_cpu = np.zeros((C, H, W, N), dtype=np.float32)
        W_cpu = np.zeros((C, R, S, K), dtype=np.float32)
        O_cpu = np.zeros((H * W * K, N), dtype=np.float32)
        
        # copy I and W from cuda to cpu
        cuda.memcpy_dtoh(I_cpu, Igpudata)
        cuda.memcpy_dtoh(W_cpu, Fgpudata)

        # create cl buffers
        I_cl = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=I_cpu)
        W_cl = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=W_cpu)
        O_cl = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=O_cpu)

        # create dummy one for bsum for now?
        bsum_cpu = np.zeros((1,), dtype=np.float32)
        bsum_cl = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=bsum_cpu)

        blockDim = len(block)
        print('blockDim', blockDim)
        if blockDim == 3:
            globalSize = (block[0] * grid[0], block[1] * grid[1], block[2] * grid[2])  # hacky? what do you mean? :-P
        else:
            raise Exception('not implemented')

        # run the conv ???
        self.kernel.conv_fprop(
            q,
            globalSize, block,
                           np.float32(alpha), np.float32(beta),
                           I_cl,
                           W_cl,
                           O_cl,
                           bsum_cl,
                           np.int32(C), np.int32(D), np.int32(H), np.int32(W), np.int32(N),
                           np.int32(T), np.int32(R), np.int32(S), np.int32(K),
                           np.int32(M), np.int32(P), np.int32(Q),
                           np.int32(str_w), np.int32(str_h), np.int32(pad_w), np.int32(pad_h),
                           np.int32(HWN), np.int32(KRST), np.int32(PQN), np.int32(PQ),
                           np.int32(zeroa), np.int32(zerob),
                           np.uint32(magic_PQ), np.uint32(shift_PQ),
                           np.uint32(magic_Q), np.uint32(shift_Q),
                           np.uint32(magic_S), np.uint32(shift_S)
        )
        
        # copy the result back...
        # first to cpu...
        cl.enqueue_copy(q, O_cpu, O_cl)

        # then to cuda...
        cuda.memcpy_htod(Ogpudata, O_cpu)

