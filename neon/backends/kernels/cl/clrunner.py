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
#        self.dummy_kernel = cl.Program(ctx, """
#kernel void copyStuff(int outSize, global float *in, global float *out) {
#    if(get_global_id(0) < outSize) {
#        out[get_global_id(0)] = in[get_global_id(0)];
#    }
#}
#""").build()

    def execute_fprop(self, grid, block, stream, alpha, beta, Igpudata, Fgpudata, Ogpudata, bsum_gpudata,
        C, D, H, W, N, T, R, S, K, M, P, Q,
        str_w, str_h, pad_w, pad_h, HWN, KRST, PQN,
        PQ, zeroa, zerob, magic_PQ, shift_PQ, magic_Q, shift_Q, magic_S, shift_S,
        *args, shared_size):
#        print('grid', grid, 'block', block, 'stream', stream, 'alpha', alpha, 'beta', beta,
#              'Igpudata', Igpudata, 'Fgpudata', Fgpudata,
#              'Ogpudata', Ogpudata, 'bsum_gpudata', bsum_gpudata)
#        print('C', C, 'D', D, 'H', H, 'W', W, 'N', N, 'T', T, 'R', R, 'S', S, 'K', K, 'M', M, 'P', P, 'Q', Q)
#        print('str_w', str_w, 'str_h', str_h, 'pad_w', pad_w, 'pad_h', pad_h, 'HWN', HWN, 'KRST', KRST, 'PQN', PQN)
#        print('PQ', PQ, 'zeroa', zeroa, 'zerob', zerob)
#        print('magic_PQ', magic_PQ, 'shift_PQ', shift_PQ, 'magic_Q', magic_Q, 'shift_Q', shift_Q)
#        print('magic_S', magic_S, 'shift_S', shift_S)
#        print('args', *args, 'shared_size', shared_size)
#        print('cuda_buffer', cuda_buffer)
#         cpu_buffer = np.zeros(
#        cpu_buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a)
        I_cpu = np.zeros((C, H, W, N), dtype=np.float32)
        W_cpu = np.zeros((C, R, S, K), dtype=np.float32)
        O_cpu = np.zeros((H * W * K, N), dtype=np.float32)
        
        # copy I and W from cuda to cpu
        cuda.Context.synchronize()
        cuda.memcpy_dtoh(I_cpu, Igpudata)
        cuda.memcpy_dtoh(W_cpu, Fgpudata)
#        cuda.Context.synchronize()

        # create cl buffers
        I_cl = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=I_cpu)
        W_cl = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=W_cpu)
        O_cl = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=O_cpu)

        # create dummy one for bsum for now?
        bsum_cpu = np.zeros((1,), dtype=np.float32)
        bsum_cl = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=bsum_cpu)
#        q.finish()

        blockDim = len(block)
#        print('blockDim', blockDim)
        if blockDim == 3:
            globalSize = (block[0] * grid[0], block[1] * grid[1], block[2] * grid[2])  # hacky? what do you mean? :-P
        else:
            raise Exception('not implemented')
#        print('globalSize', globalSize)
        
#        outSize = H*W*K*N
#        assert outSize < pow(2, 30)
#        print('outSize', outSize)
#        roundedOutSize = (outSize // 256) * 256
#        self.dummy_kernel.copyStuff(q, (roundedOutSize,), (256,), np.int32(outSize),
#            I_cl, O_cl
#        )

        # run the conv ???
        q.finish()
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
        q.finish()
        start = time.time()
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
        q.finish()
        end = time.time()
        print('kernel wallclock time', (end-start))
        cl.enqueue_copy(q, O_cpu, O_cl)
#        q.finish()

        # then to cuda...
        cuda.memcpy_htod(Ogpudata, O_cpu)
#        cuda.Context.synchronize()

    def execute_bprop(self, grid, block, stream, alpha, beta, Igpudata, filtertemp_gpudata, Ogpudata, bsum_gpudata,
        K, M, P, Q, N, T, R, S, C, D, H, W,
             str_w, str_h, pad_w, pad_h,
             PQN, CRST, HWN,
             HW, zeroa, zerob,
             magic_HW, shift_HW, magic_W, shift_W, magic_S, shift_S,
        *args, shared_size):
#        print('grid', grid, 'block', block, 'stream', stream, 'alpha', alpha, 'beta', beta,
#              'Igpudata', Igpudata, 'Fgpudata', Fgpudata,
#              'Ogpudata', Ogpudata, 'bsum_gpudata', bsum_gpudata)
#        print('C', C, 'D', D, 'H', H, 'W', W, 'N', N, 'T', T, 'R', R, 'S', S, 'K', K, 'M', M, 'P', P, 'Q', Q)
#        print('str_w', str_w, 'str_h', str_h, 'pad_w', pad_w, 'pad_h', pad_h, 'HWN', HWN, 'KRST', KRST, 'PQN', PQN)
#        print('PQ', PQ, 'zeroa', zeroa, 'zerob', zerob)
#        print('magic_PQ', magic_PQ, 'shift_PQ', shift_PQ, 'magic_Q', magic_Q, 'shift_Q', shift_Q)
#        print('magic_S', magic_S, 'shift_S', shift_S)
        print('args', *args, 'shared_size', shared_size)
#        print('cuda_buffer', cuda_buffer)
#         cpu_buffer = np.zeros(
#        cpu_buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a)
        I_cpu = np.zeros((C, H, W, N), dtype=np.float32)
        filtertemp_cpu = np.zeros((K * R * S * C,), dtype=np.float32)
        O_cpu = np.zeros((H * W * K, N), dtype=np.float32)
        
        # copy I and W from cuda to cpu
        cuda.Context.synchronize()
        cuda.memcpy_dtoh(I_cpu, Igpudata)
        cuda.memcpy_dtoh(filtertemp_cpu, filtertemp_gpudata)
#        cuda.Context.synchronize()

        # create cl buffers
        I_cl = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=I_cpu)
        filtertemp_cl = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=filtertemp_cpu)
        O_cl = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=O_cpu)

        # create dummy one for bsum for now?
        bsum_cpu = np.zeros((1,), dtype=np.float32)
        bsum_cl = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=bsum_cpu)
#        q.finish()

        blockDim = len(block)
#        print('blockDim', blockDim)
        if blockDim == 3:
            globalSize = (block[0] * grid[0], block[1] * grid[1], block[2] * grid[2])  # hacky? what do you mean? :-P
        else:
            raise Exception('not implemented')
#        print('globalSize', globalSize)
        
#        outSize = H*W*K*N
#        assert outSize < pow(2, 30)
#        print('outSize', outSize)
#        roundedOutSize = (outSize // 256) * 256
#        self.dummy_kernel.copyStuff(q, (roundedOutSize,), (256,), np.int32(outSize),
#            I_cl, O_cl
#        )

        q.finish()
        start = time.time()
        self.kernel.conv_bprop(
            q,
            globalSize, block,
                           np.float32(alpha), np.float32(beta),
                           I_cl,
                           filtertemp_cl,
                           O_cl,
                           bsum_cl,
                           
                           np.int32(K), np.int32(M), np.int32(P), np.int32(Q), np.int32(N), np.int32(T), np.int32(R), np.int32(S), np.int32(C), np.int32(D), np.int32(H), np.int32(W),
                           np.int32(str_w), np.int32(str_h), np.int32(pad_w), np.int32(pad_h),
                           np.int32(PQN), np.int32(CRST), np.int32(HWN),
                           np.int32(HW), np.int32(zeroa), np.int32(zerob),
                           
                           np.uint32(magic_HW), np.uint32(shift_HW), np.uint32(magic_W), np.uint32(shift_W), np.uint32(magic_S), np.uint32(shift_S)
        )
        q.finish()
        end = time.time()
        print('kernel wallclock time', (end-start))
        cl.enqueue_copy(q, O_cpu, O_cl)
#        q.finish()

        # then to cuda...
        cuda.memcpy_htod(Ogpudata, O_cpu)
#        cuda.Context.synchronize()

