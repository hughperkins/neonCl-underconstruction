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
import pycuda.driver as drv
import sys
from pycuda.tools import context_dependent_memoize
from neon.backends.cuda_templates import _ew_types
from neon.backends.util.math_helper import magic64, magic32, ceil_div
from neon.backends.kernels.cl import convolution_cl
from neon.backends.kernels.cl.clrunner import ClRunner, ShuffleRunner


if sys.version_info >= (3, 0):
    from functools import reduce


class KernelGroup(object):
    def __init__(self, lib, dtype):
        self.lib = lib
        self.dtype = dtype
        self.dtype_str = dtype.str[1:]
        self.vec_size = 4 if dtype.itemsize == 4 else 8

        if dtype.type is np.float32:
            self.clss = "sconv"
        else:
            raise TypeError("dtype not supported.")

    def __str__(self):
        raise TypeError("please implement __str__ to describe kernel params for logging.")

    def bind_params(self, *args):
        raise TypeError("bind_params not implemented.")

    def execute(self, repeat=1, unbind=True):
        raise TypeError("execute not implemented.")

    def init_bsum(self, bsum, flags):
        flags |= self.flags
        if bsum:
            bsum_gpudata = bsum.gpudata
            self.bsum_zero = [bsum_gpudata, 0, bsum.size, self.lib.stream]
            flags |= 4
        else:
            bsum_gpudata = 0
            self.bsum_zero = 0
            flags &= ~4
        return bsum_gpudata, flags


class FpropCuda(KernelGroup):
    def __init__(self, lib, dtype,
                 N, C, K,
                 D, H, W,
                 T, R, S,
                 M, P, Q,
                 pad_d, pad_h, pad_w,
                 str_d, str_h, str_w,
                 bsum):

        super(FpropCuda, self).__init__(lib, dtype)

        assert N % 32 == 0, "N dim must be multiple of 32"
        assert K % self.vec_size == 0, "K dim must be multiple of %d" % self.vec_size

        magic_PQ = magic64(P*Q)
        magic_Q = magic64(Q)
        magic_S = magic32(R*S+32, S)
        HWN = H * W * N
        RST = R * S * T
        KRST = K * RST
        PQ = P * Q
        PQN = PQ * N
        self.clRunner = ClRunner(ctx=self.lib.cl_ctx, q=self.lib.q, dtype=self.dtype.str[1:], filter_size=R*S,
                                       bsum=bsum, operation="fprop")
        grid = (PQ * (-(-N // 32)), (-(-K // 32)), 1)
        block = (8, 8, 1)
        static_kernel_args = _flatten([C, D, H, W, N, T, R, S, K, M, P, Q,
                                       str_w, str_h, pad_w, pad_h,
                                       HWN // 4, KRST // 4, PQN // 4,
                                       PQ, 0, 0,
                                       magic_PQ, magic_Q, magic_S])
        self.launch_args = [grid, block] + [None] * 7 + static_kernel_args

        self.shared = RST * 4 * 2
        self.flags = (bsum and 4)

    def bind_params(self, I, F, O, alpha, beta, bsum, flags=0):
        assert I.dtype == F.dtype == O.dtype
        bsum_gpudata, flags = self.init_bsum(bsum, flags)
        self.launch_args[2:9] = (self.lib.stream, alpha, beta,
                                 I.gpudata, F.gpudata, O.gpudata, bsum_gpudata)

    def execute(self, repeat=1, unbind=True):
#        print('repeat', repeat)
        for r in range(repeat):
            if self.bsum_zero:
                drv.memset_d32_async(*self.bsum_zero)
#            print('calling kernel', self.kernel, 'args', self.launch_args, 'shared_size', self.shared)
#            self.kernel.prepared_async_call(*self.launch_args, shared_size=self.shared)
            self.clRunner.execute_fprop(*self.launch_args, shared_size=self.shared)
        if unbind:
            self.bsum_zero = None
            self.launch_args[2:9] = (None,) * 7

    def __str__(self):
        return "FpropCuda"


class BpropCuda(KernelGroup):
    def __init__(self, lib, dtype,
                 N, C, K,
                 D, H, W,
                 T, R, S,
                 M, P, Q,
                 pad_d, pad_h, pad_w,
                 str_d, str_h, str_w,
                 bsum):

        super(BpropCuda, self).__init__(lib, dtype)

        assert N % 32 == 0, "N dim must be multiple of 32"
        assert K % self.vec_size == 0, "K dim must be multiple of %d" % self.vec_size

        magic_HW = magic64(H*W)
        magic_W = magic64(W)
        magic_RS = magic32(R*S*T+32, R*S)
        magic_S = magic32(R*S+32, S)
        HW = H * W
        HWN = HW * N
        RST = R * S * T
        CRST = C * RST
        PQ = P * Q
        PQN = PQ * N

        self.bsum = bsum
#        self.kernel = _get_conv_kernel(dtype=self.dtype.str[1:], filter_size=R*S,
#                                       bsum=bsum, operation="bprop")
        self.clRunner = ClRunner(ctx=self.lib.cl_ctx, q=self.lib.q, dtype=self.dtype.str[1:], filter_size=R*S,
                                       bsum=bsum, operation="bprop")
        grid = (HW * (-(-N // 32)), -(-C // 32), 1)
        block = (8, 8, 1)
        static_kernel_args = _flatten([K, M, P, Q, N, T, R, S, C, D, H, W,
                                       str_w, str_h, pad_w, pad_h,
                                       PQN // 4, CRST // 4, HWN // 4,
                                       HW, 0, 0,
                                       magic_HW, magic_W, magic_S])
        self.launch_args = [grid, block] + [None] * 7 + static_kernel_args

        self.shared = R*S*T * 4 * 2
        self.flags = (bsum and 4)

        # generate the kernel args for dim shuffling CTRSK => KTRSC
        shuffle_grid = (ceil_div(K, 32), ceil_div(C, 32), R*S*T)
        self.shuffle_size = C*T*R*S*K*dtype.itemsize
        self.shuffle_args = [shuffle_grid, (32, 8, 1), None, None, None]
        self.shuffle_args.extend(_flatten([
            R*S*T*K, R*S*K, S*K, K,
            R*S*T*C, R*S*C, S*C, C,
            R*S, T, R, S, magic_RS, magic_S]))

        lib.set_scratch_size(self.shuffle_size)
        self.shuffleRunner = ShuffleRunner(ctx=self.lib.cl_ctx, q=self.lib.q, dtype=self.dtype)

    def bind_params(self, gradO, W, gradI, alpha, beta, bsum, flags=0):
        assert gradO.dtype == W.dtype == gradI.dtype
        if self.bsum:
            assert bsum is not None, "must use initialized bsum config"

        bsum_gpudata, flags = self.init_bsum(bsum, flags)

        Wt = self.lib.scratch_buffer(self.shuffle_size)

        self.shuffle_args[2:5] = (self.lib.stream, Wt, W.gpudata)
        self.launch_args[2:9] = (self.lib.stream, alpha, beta,
                                 gradO.gpudata, Wt, gradI.gpudata, bsum_gpudata)

    def execute(self, repeat=1, unbind=True):
        C = self.shuffle_args[12]
        assert C >= 4, "C dim must be 4 or greater for CUDA C backprop kernel"
        for r in range(repeat):
            if self.bsum_zero:
                drv.memset_d32_async(*self.bsum_zero)
            self.shuffleRunner.execute(*self.shuffle_args)
            self.clRunner.execute_bprop(*self.launch_args, shared_size=self.shared)
        if unbind:
            self.bsum_zero = None
            self.shuffle_args[2:5] = (None,) * 3
            self.launch_args[2:9] = (None,) * 7

    def __str__(self):
        return "BpropCuda"


class UpdateCuda(KernelGroup):
    def __init__(self, lib, dtype,
                 N, C, K,
                 D, H, W,
                 T, R, S,
                 M, P, Q,
                 pad_d, pad_h, pad_w,
                 str_d, str_h, str_w):

        super(UpdateCuda, self).__init__(lib, dtype)

        assert N % 32 == 0, "N dim must be multiple of 32"

        HWN = H * W * N
        RS = R * S
        RST = RS * T
        KRST = K * RST
        CRSTK = KRST * C
        PQ = P * Q
        PQN = PQ * N
        magic_S = magic32(R*S+32, S)

        if lib.deterministic:
            grid_P = 1
            grid_Q = 1
            self.determ = CRSTK
        else:
            grid_P = P
            grid_Q = Q
            self.determ = 0

        pq_blocks = grid_P * grid_Q
        magic_PQ = magic64(pq_blocks)
        magic_Q = magic64(grid_Q)

#        self.kernel = _get_conv_kernel(dtype=self.dtype.str[1:], filter_size=R*S,
#                                       bsum=False, operation="update")
        self.clRunner = ClRunner(ctx=self.lib.cl_ctx, q=self.lib.q, dtype=self.dtype.str[1:], filter_size=R*S,
                                       bsum=False, operation="update")
        grid = (pq_blocks * (-(-K // 32)), (-(-(C*RS) // 32)), 1)
        block = (8, 32, 1)
        static_kernel_args = _flatten([C, D, H, W, N, T, R, S, K, M, P, Q,
                                       str_w, str_h, pad_w, pad_h,
                                       HWN // 4, KRST // 4, PQN // 4,
                                       PQ, grid_P, grid_Q,
                                       magic_PQ, magic_Q, magic_S])
        self.launch_args = [grid, block] + [None] * 7 + static_kernel_args

        lib.set_scratch_size((self.determ or C*T*R*S*K)*4)

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
        assert I.dtype == gradO.dtype
        if gradW.dtype is not np.float32:
            update_temp = self.lib.scratch_buffer((self.determ or gradW.size)*4)
            self.convert_args = [update_temp, "f4", gradW, False]
        else:
            update_temp = gradW.gpudata
            self.convert_args = False

        self.zero_args = [update_temp, 0, gradW.size, self.lib.stream]

        beta = 0.0
        bsum_gpudata = 0
        self.launch_args[2:9] = (self.lib.stream, alpha, beta,
                                 I.gpudata, gradO.gpudata, gradW.gpudata, bsum_gpudata)

    def execute(self, repeat=1, unbind=True):
        for r in range(repeat):
            cl.enqueue_fill_buffer(self.lib.q, self.zero_args[0], np.float32(0), 0, self.zero_args[2] * 4)
            self.clRunner.execute_update(*self.launch_args)
            if self.convert_args:
                _fp_convert(*self.convert_args)

        if unbind:
            self.zero_args = self.convert_args = None
            self.launch_args[2:9] = (None,) * 7

    def __str__(self):
        return "UpdateCuda"


# flatten a nested list of lists or values
def _flatten(lst):
    return sum(([x] if not isinstance(x, (list, tuple))
                else _flatten(x) for x in lst), [])

