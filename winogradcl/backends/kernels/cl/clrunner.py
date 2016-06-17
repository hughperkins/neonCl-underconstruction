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

# import time
import numpy as np
import pyopencl as cl
from winogradcl.backends.cuda_templates import _ew_types
from winogradcl.backends.kernels.cl import convolution_cl
from winogradcl.backends.kernels.cl.callkernel import call_cl_kernel


mf = cl.mem_flags

class ClRunner(object):
    def __init__(self, ctx, q, dtype, filter_size, operation):
        self.ctx = ctx
        self.q = q
        self.dtype = dtype
        self.filter_size = filter_size
        self.operation = operation
        self.kernel = convolution_cl._get_conv_kernel(
            ctx=ctx, options='', dtype=self.dtype, filter_size=self.filter_size,
            operation=self.operation)

    def execute_fprop(self, *args):
        call_cl_kernel(self.kernel.conv_fprop,
            self.q,
            *args
        )

    def execute_bprop(self, *args):
        call_cl_kernel(self.kernel.conv_bprop,
            self.q,
            *args
        )

    def execute_update(self, *args):
        call_cl_kernel(self.kernel.conv_update,
            self.q,
            *args
        )

