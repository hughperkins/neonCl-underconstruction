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

import time
import numpy as np
import pyopencl as cl
from winogradcl.backends.cuda_templates import _ew_types
from winogradcl.backends.kernels.cl import convolution_cl
from winogradcl.backends.kernels.cl.callkernel import call_cl_kernel


mf = cl.mem_flags

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
        *args, shared_size):

        # create dummy one for bsum for now?
        bsum_cpu = np.zeros((1,), dtype=np.float32)
        bsum_cl = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=bsum_cpu)

        blockDim = len(block)
        if blockDim == 3:
            globalSize = (block[0] * grid[0], block[1] * grid[1], block[2] * grid[2])  # hacky? what do you mean? :-P
        else:
            raise Exception('not implemented')

        call_cl_kernel(self.kernel.conv_fprop,
            self.q, grid, block,
            alpha, beta,
            I_cl,
            W_cl,
            O_cl,
            bsum_cl,
            *args
        )

    def execute_bprop(self, grid, block, stream, alpha, beta, 
            gradO_cl,
            Wt_cl,
            gradI_cl,
            bsum_gpudata,
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
            *args
        )

    def execute_update(
            self, grid, block, stream, alpha, beta,
            I_cl,
            gradO_cl,
            gradW_cl,
            bsum_gpudata,
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
            *args
        )

