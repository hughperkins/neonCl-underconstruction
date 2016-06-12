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
from neon.backends.cuda_templates import _ew_types
from neon.backends.kernels.cl import convolution_cl

mf = cl.mem_flags

def call_cl_kernel(kernel, queue, grid, block, *args):
    blockDim = len(block)
    if blockDim == 3:
        globalSize = (block[0] * grid[0], block[1] * grid[1], block[2] * grid[2])  # hacky? what do you mean? :-P
    else:
        raise Exception('not implemented')

    newargs = []
    i = 0
    for arg in args:
        if isinstance(arg, int):
            newargs.append(np.int32(arg))
        elif isinstance(arg, float):
            newargs.append(np.float32(arg))
        elif isinstance(arg, cl.cffi_cl.Buffer):
            newargs.append(arg)
        else:
            raise Exception('type not implemented %s' % type(arg))
        i += 1
    kernel(queue, globalSize, block, *newargs)

class ShuffleRunner(object):
    def __init__(self, ctx, q, dtype):
        self.ctx = ctx
        self.q = q
        self.dtype = dtype
        self.shuffle_kernel_cl = _get_shuffle_kernel_cl(self.ctx, dtype.str[1:])
        
    def execute(self, grid, block, stream, filtertemp_cl, F_cl,
            RSTK, RSK, SK, K, RSTC, RSC, SC, C,
            RS, T, R, S, magic_RS, shift_RS, magic_S, shift_S):

        call_cl_kernel(self.shuffle_kernel_cl.dimShuffle,
            self.q, grid, block, 
            filtertemp_cl,
            F_cl,
            RSTK, RSK, SK, K, RSTC, RSC, SC, C,
            RS, T, R, S, magic_RS, shift_RS, magic_S, shift_S
        )


class ClRunner(object):
    def __init__(self, ctx, q, dtype, filter_size, bsum, operation):
        self.ctx = ctx
        self.q = q
        self.dtype = dtype
        self.filter_size = filter_size
        self.bsum = bsum
        self.operation = operation
        self.kernel = convolution_cl._get_conv_kernel(
            ctx=ctx, options='', dtype=self.dtype, filter_size=self.filter_size,
            bsum=self.bsum, operation=self.operation)

    def execute_fprop(self, grid, block, stream, alpha, beta, I_cl, W_cl, O_cl, bsum_gpudata,
        C, D, H, W, N, T, R, S, K, M, P, Q,
        str_w, str_h, pad_w, pad_h, HWN, KRST, PQN,
        PQ, zeroa, zerob, magic_PQ, shift_PQ, magic_Q, shift_Q, magic_S, shift_S,
        *args, shared_size):

        # create dummy one for bsum for now?
        bsum_cpu = np.zeros((1,), dtype=np.float32)
        bsum_cl = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=bsum_cpu)

        blockDim = len(block)
        if blockDim == 3:
            globalSize = (block[0] * grid[0], block[1] * grid[1], block[2] * grid[2])  # hacky? what do you mean? :-P
        else:
            raise Exception('not implemented')

        self.q.finish()
        self.kernel.conv_fprop(
            self.q,
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

    def execute_bprop(self, grid, block, stream, alpha, beta, 
            gradO_cl,
            Wt_cl,
            gradI_cl,
            bsum_gpudata,            
            K, M, P, Q, N, T, R, S, C, D, H, W,
            *args, shared_size):

        # create dummy one for bsum for now?
        bsum_cpu = np.zeros((1,), dtype=np.float32)
        bsum_cl = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=bsum_cpu)

        call_cl_kernel(self.kernel.conv_bprop,
            self.q, grid, block,
            alpha, beta,
            gradO_cl,
            Wt_cl,
            gradI_cl,
            bsum_cl,
            K, M, P, Q, N, T, R, S, C, D, H, W,
            *args
        )

    def execute_update(
            self, grid, block, stream, alpha, beta,
            I_cl,
            gradO_cl,
            gradW_cl,
            bsum_gpudata,
            C, D, H, W, N, T, R, S, K, M, P, Q,
            *args):

        # create dummy one for bsum for now?
        bsum_cpu = np.zeros((1,), dtype=np.float32)
        bsum_cl = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=bsum_cpu)

        call_cl_kernel(self.kernel.conv_update,
            self.q, grid, block, 
            alpha, beta,
            I_cl,
            gradO_cl,
            gradW_cl,
            bsum_cl,
            C, D, H, W, N, T, R, S, K, M, P, Q,
            *args
        )


def _get_shuffle_kernel_cl(ctx, dtype):
    _shuffle_kernel = r"""
kernel void dimShuffle(
    global %(type)s* out, global const %(type)s* in,
    int TRSK, int RSK, int SK, int K,
    int TRSC, int RSC, int SC, int C,
    int RS, int T, int R, int S,
    int magic_RS, int shift_RS,
    int magic_S,  int shift_S)
{
    local %(type)s tile[32][33];

    int tx  = get_local_id(0);
    int ty  = get_local_id(1);
    int bk  = get_group_id(0);
    int bc  = get_group_id(1);
    int trs = get_group_id(2);

    int k  = bk * 32 + tx;
    int c  = bc * 32 + ty;

    int t  = magic_RS * trs; t >>= shift_RS;
    int rs = trs - t*RS;

    int r = magic_S * rs; r >>= shift_S;
    int s = rs - r*S;

    for (int j = 0; j < 32; j += 8)
    {
        int cj = c + j;
        if (cj < C && k < K)
            tile[ty + j][tx] = in[ cj*TRSK + t*RSK + r*SK + s*K + k ];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    k = bk * 32 + ty;
    c = bc * 32 + tx;

    // Mirror RST
    s = S - s - 1;
    r = R - r - 1;
    t = T - t - 1;

    for (int i = 0; i < 32; i += 8)
    {
        int ki = k + i;
        if (ki < K && c < C)
            out[ ki*TRSC + t*RSC + r*SC + s*C + c ] = tile[tx][ty + i];
    }
}
"""
    code = _shuffle_kernel % _ew_types[dtype]
    module = cl.Program(ctx, code).build()
    return module

