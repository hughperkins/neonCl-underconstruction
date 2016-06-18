# Copyright 2014 Nervana Systems Inc., 2016 Hugh Perkins, All rights reserved.
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
Python code to wrap convolution kernels
"""

import numpy as np
import pyopencl as cl
import sys
from neoncl.backends.cuda_templates import _ew_types
from neoncl.util.math_helper import get_div_mul_shift_32, get_div_mul_shift_64, ceil_div
from neoncl.backends.kernels.cl import convolution_cl
from neoncl.backends.kernels.cl.clshuffler import get_shuffle_kernel_cl
from neoncl.backends.kernels.cl.callkernel import call_cl_kernel


if sys.version_info >= (3, 0):
    from functools import reduce


class KernelGroup(object):
    def __init__(self, dtype):
        self.vec_size = 4
        self.clss = 'sconv'
        self.dtype = dtype

    def __str__(self):
        raise TypeError("please implement __str__ to describe kernel params for logging.")

    def bind_params(self, *args):
        raise TypeError("bind_params not implemented.")

    def execute(self, repeat=1, unbind=True):
        raise TypeError("execute not implemented.")


#    N: Number of images in mini-batch
#    C: Number of input feature maps
#    K: Number of output feature maps

#    D: Depth  of input image
#    H: Height of input image
#    W: Width  of input image

#    T: Depth  of filter kernel
#    R: Height of filter kernel
#    S: Width  of filter kernel

#    M: depth of output
#    P: height of output
#    Q: width of output

class FpropCuda(KernelGroup):
    def __init__(self, ctx, dtype,
                 N, C, K,
                 D, H, W,
                 T, R, S,
                 M, P, Q,
                 pad_d, pad_h, pad_w,
                 str_d, str_h, str_w
                 ):

        super(FpropCuda, self).__init__(dtype)

        assert N % 32 == 0, "N dim must be multiple of 32"
        assert K % self.vec_size == 0, "K dim must be multiple of %d" % self.vec_size

        div_PQ_mul_shift = get_div_mul_shift_64(P*Q)
        div_Q_mul_shift = get_div_mul_shift_64(Q)
        div_S_mul_shift = get_div_mul_shift_32(R*S+32, S)
        HWN = H * W * N
        RST = R * S * T
        KRST = K * RST
        PQ = P * Q
        PQN = PQ * N
        self.kernel = convolution_cl._get_conv_kernel(
            ctx=ctx, options='', dtype=self.dtype, filter_size=R*S,
            operation='fprop')
        grid = (PQ * (-(-N // 32)), (-(-K // 32)), 1)
        block = (8, 8, 1)
        static_kernel_args = _flatten([C, D, H, W, N, T, R, S, K, M, P, Q,
                                       str_w, str_h, pad_w, pad_h,
                                       HWN // 4, KRST // 4, PQN // 4,
                                       PQ, 0, 0,
                                       div_PQ_mul_shift, div_Q_mul_shift, div_S_mul_shift])
        self.launch_args = [grid, block] + [None] * 5 + static_kernel_args

        self.shared = RST * 4 * 2

    def bind_params(self, I, F, O, alpha, beta, flags=0):
        self.launch_args[2:7] = (alpha, beta,
                                 I, F, O)

    def execute(self, q, repeat=1, unbind=True):
        for r in range(repeat):
            call_cl_kernel(self.kernel, q, *self.launch_args)
        if unbind:
            self.launch_args[2:7] = (None,) * 5

    def __str__(self):
        return "FpropCuda"


class BpropCuda(KernelGroup):
    def __init__(self, ctx, dtype,
                 N, C, K,
                 D, H, W,
                 T, R, S,
                 M, P, Q,
                 pad_d, pad_h, pad_w,
                 str_d, str_h, str_w
                 ):

        super(BpropCuda, self).__init__(dtype)

        assert N % 32 == 0, "N dim must be multiple of 32"
        assert K % self.vec_size == 0, "K dim must be multiple of %d" % self.vec_size

        div_HW_mul_shift = get_div_mul_shift_64(H*W)
        div_W_mul_shift = get_div_mul_shift_64(W)
        div_RS_mul_shift = get_div_mul_shift_32(R*S*T+32, R*S)
        div_S_mul_shift = get_div_mul_shift_32(R*S+32, S)
        HW = H * W
        HWN = HW * N
        RST = R * S * T
        CRST = C * RST
        PQ = P * Q
        PQN = PQ * N

        self.kernel = convolution_cl._get_conv_kernel(
            ctx=ctx, options='', dtype=self.dtype, filter_size=R*S,
            operation='bprop')
        grid = (HW * (-(-N // 32)), -(-C // 32), 1)
        block = (8, 8, 1)
        static_kernel_args = _flatten([K, M, P, Q, N, T, R, S, C, D, H, W,
                                       str_w, str_h, pad_w, pad_h,
                                       PQN // 4, CRST // 4, HWN // 4,
                                       HW, 0, 0,
                                       div_HW_mul_shift, div_W_mul_shift, div_S_mul_shift])
        self.launch_args = [grid, block] + [None] * 5 + static_kernel_args

        self.shared = R*S*T * 4 * 2

        # generate the kernel args for dim shuffling CTRSK => KTRSC
        dtype_itemsize = 4
        shuffle_grid = (ceil_div(K, 32), ceil_div(C, 32), R*S*T)
        self.shuffle_size = C*T*R*S*K*dtype_itemsize
        self.shuffle_args = [shuffle_grid, (32, 8, 1), None, None]
        self.shuffle_args.extend(_flatten([
            R*S*T*K, R*S*K, S*K, K,
            R*S*T*C, R*S*C, S*C, C,
            R*S, T, R, S, div_RS_mul_shift, div_S_mul_shift]))

        # lib.set_scratch_size(self.shuffle_size)
        self.shuffleKernel = get_shuffle_kernel_cl(ctx, dtype)

    def shuffle(self, q, Wt, W):
        self.shuffle_args[2:4] = (Wt, W)
        call_cl_kernel(self.shuffleKernel, q, *self.shuffle_args)

    def bind_params(self, gradO, Wt, gradI, alpha, beta, flags=0):
        # Wt = self.lib.scratch_buffer(self.shuffle_size)

        #self.shuffle_args[2:4] = (Wt, W.gpudata)
        self.launch_args[2:7] = (alpha, beta,
                                 gradO, Wt, gradI)

    def execute(self, q, repeat=1, unbind=True):
        C = self.shuffle_args[12]
        assert C >= 4, "C dim must be 4 or greater for CUDA C backprop kernel"
        for r in range(repeat):
            # call_cl_kernel(self.shuffleKernel, self.lib.q, *self.shuffle_args)
            call_cl_kernel(self.kernel, q, *self.launch_args)
        if unbind:
            # self.shuffle_args[2:4] = (None,) * 2
            self.launch_args[2:7] = (None,) * 5

    def __str__(self):
        return "BpropCuda"


class UpdateCuda(KernelGroup):
    def __init__(self, ctx, dtype,
                 N, C, K,
                 D, H, W,
                 T, R, S,
                 M, P, Q,
                 pad_d, pad_h, pad_w,
                 str_d, str_h, str_w):

        super(UpdateCuda, self).__init__(dtype)

        assert N % 32 == 0, "N dim must be multiple of 32"

        HWN = H * W * N
        RS = R * S
        RST = RS * T
        KRST = K * RST
        CRSTK = KRST * C
        PQ = P * Q
        PQN = PQ * N
        div_S_mul_shift = get_div_mul_shift_32(R*S+32, S)

        self.W_size = C * K * R * S

        #if lib.deterministic:
        #    grid_P = 1
        #    grid_Q = 1
        #    self.determ = CRSTK
        #else:
        grid_P = P
        grid_Q = Q
        # self.determ = 0

        pq_blocks = grid_P * grid_Q
        div_PQ_mul_shift = get_div_mul_shift_64(pq_blocks)
        div_Q_mul_shift = get_div_mul_shift_64(grid_Q)

        self.kernel = convolution_cl._get_conv_kernel(
            ctx=ctx, options='', dtype=dtype, filter_size=R*S,
            operation='update')
        grid = (pq_blocks * (-(-K // 32)), (-(-(C*RS) // 32)), 1)
        block = (8, 32, 1)
        static_kernel_args = _flatten([C, D, H, W, N, T, R, S, K, M, P, Q,
                                       str_w, str_h, pad_w, pad_h,
                                       HWN // 4, KRST // 4, PQN // 4,
                                       PQ, grid_P, grid_Q,
                                       div_PQ_mul_shift, div_Q_mul_shift, div_S_mul_shift])
        self.launch_args = [grid, block] + [None] * 5 + static_kernel_args

        # lib.set_scratch_size((determ or C*T*R*S*K)*4)

    def update_grid(self, kernel_name, base_blocks, P, Q, SM_count):
        threads = kernel_specs.kernels[kernel_name]["threads"]
        occupancy = kernel_specs.kernels[kernel_name]["occupancy"]

        # warps per scheduler for one block
        occ_per_block = threads / (32.0 * 4.0 * SM_count)

        grid = []
        for p in range(1, P+1):
            for q in range(1, Q+1):

                occup = p*q*base_blocks * occ_per_block
                groups = occup / occupancy
                slots = ceil(groups)

                # This is a heuristic that keeps the balance of work accross the SMs
                # while also maximizing the work that each block does
                heuristic = min(abs(x - slots) for x in range(4, 8)) + (slots - groups) / 100.0

                grid.append((p, q, heuristic))

        grid.sort(key=lambda x: x[-1])

        return (grid[0][0], grid[0][1], threads)

    def bind_params(self, I, gradO, gradW, alpha):
        self.zero_args = [gradW, 0, self.W_size]

        beta = 0.0
        self.launch_args[2:7] = (alpha, beta,
                                 I, gradO, gradW)

    def execute(self, q, repeat=1, unbind=True):
        for r in range(repeat):
            cl.enqueue_fill_buffer(q, self.zero_args[0], np.float32(0), 0, self.zero_args[2] * 4)
            call_cl_kernel(self.kernel, q, *self.launch_args)

        if unbind:
            self.zero_args = self.convert_args = None
            self.launch_args[2:7] = (None,) * 5

    def __str__(self):
        return "UpdateCuda"


# flatten a nested list of lists or values
def _flatten(lst):
    return sum(([x] if not isinstance(x, (list, tuple))
                else _flatten(x) for x in lst), [])

