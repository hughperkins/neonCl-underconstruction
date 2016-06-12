# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc., 2016 Hugh Perkins, All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
"""
Our GPU based backend interface and tensor data structure.
"""

import os
import sys
import numpy as np
import pycuda.driver as drv
import pyopencl as cl
#from pycuda.tools import context_dependent_memoize
import logging

from neon.backends.backend import Backend
from neon.backends.layer_gpu import ConvLayer, _get_sm_count

_none_slice = slice(None, None, None)

logger = logging.getLogger(__name__)

mf = cl.mem_flags

class NervanaGPU(Backend):
    """
    The primary interface class and factory for GPUTensors

    Arguments:
        stochastic_round (int or bool, optional): set to desired number of mantissa
                                                    bits to stochasically round to.
                                                    Set to 0 or False to disable
                                                    stochastic rounding (the default).
                                                    Set to True to use default
                                                    rounding bit width.
        bench (bool, optional): set to True to print out performance data for
                                    most kernel calls.  If False (default) no
                                    performance data is printed.
        compat_mode (str, optional): set flag to match implementation of other libraries
                                     for compatibility.  currently only 'caffe' is supported

        TODO: define other keyword parameters!
        """

    # currently this is hard wired
    def __init__(self,
                 default_dtype=np.float32,
                 stochastic_round=False,
                 deterministic=None,
                 device_id=0,
                 bench=False,
                 scratch_size=0,
                 hist_bins=64,
                 hist_offset=-48,
                 compat_mode=None,
                 enable_winograd=True,
                 cache_dir=os.path.join(os.path.expanduser('~'), 'nervana/cache')):
        if default_dtype not in [np.float16, np.float32]:
            raise ValueError('Default data type for nervanagpu '
                             'backend must be float16 or 32')

        if default_dtype is np.float32:
            if stochastic_round:
                if stochastic_round is True:
                    raise ValueError('Default rounding bit width is not '
                                     'supported for fp32.  Please specify '
                                     'number of bits to round to.')
                logger.warn('Using 32 bit floating point and setting stochastic '
                            'rounding to %d bits' % stochastic_round)

        # context
        drv.init()
        self.device_type = 1
        self.device_id = device_id if device_id is not None else 0
        self.ctx = drv.Device(device_id).make_context()

        # cl context
        platforms = cl.get_platforms()
        i = 0
        for platform in platforms:
           gpu_devices = platform.get_devices(device_type=cl.device_type.GPU)
           if self.device_id < i + len(gpu_devices):
               self.cl_ctx = cl.Context(devices=[gpu_devices[self.device_id - i]])
               break
           i += len(gpu_devices)

        print('cl_context', self.cl_ctx)
        #ctx = cl.create_some_context()
        self.q = cl.CommandQueue(self.cl_ctx)

        # super class init
        super(NervanaGPU, self).__init__(default_dtype,
                                         compat_mode=compat_mode,
                                         deterministic=deterministic)

        # log
        logger.info("Initialized NervanaGPU")

        # stochastic_round
        assert stochastic_round is False, "Are you sure about using SR globally in the backend?"
        if stochastic_round:
            if stochastic_round is True:
                stochastic_round = 10
        else:
            stochastic_round = 0

        # attributes
        self.scratch_size = scratch_size
        self.scratch_offset = 0
        self.round_mode = stochastic_round
        self.bench = bench
        self.stream = None
        self.buf = {}
        self.buf_active = {}
        self.warmup = False

        # store histograms for batched memcpy
        self.hist_bins = hist_bins
        self.hist_offset = hist_offset
        self.hist_map = dict()
        self.hist_idx = 0
        self.hist_max = 4*4096
#        self.hist_base = drv.mem_alloc(self.hist_bins * self.hist_max * 4)
#        drv.memset_d32(self.hist_base, 0, self.hist_bins * self.hist_max)

        self.compute_capability = (4,0)
        self.use_cudac_kernels = True

        self.enable_winograd = enable_winograd
        self.cache_dir = cache_dir
        if not os.path.isdir(self.cache_dir):
            os.makedirs(self.cache_dir)

    def scratch_buffer(self, size):

        if size & 127 != 0:
            size += 128 - (size & 127)

        if size > self.scratch_size:
            raise RuntimeError("nervanagpu.scratch_size(%d) is too small for this operation." % self.scratch_size)

        self.scratch_offset = size

#        return int(_get_scratch_data(self.scratch_size))
        return self.scratch

    def set_scratch_size(self, *args):
        total_size = 0
        for size in args:
            if size & 127 != 0:
                size += 128 - (size & 127)
            total_size += size

        if total_size > self.scratch_size:
            self.scratch_size = total_size
            self.scratch_cpu = np.zeros((total_size/4,), dtype=np.float32)
            self.scratch = cl.Buffer(self.cl_ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.scratch_cpu)

    def execute(self, optree):
        """
        Execute the optree. Break optree into sub-optrees if necessary.
        """
        raise Exception('not implemented')

    def conv_layer(self, dtype,
                   N, C, K,
                   D=1, H=1, W=1,
                   T=1, R=1, S=1,
                   pad_d=0, pad_h=0, pad_w=0,
                   str_d=1, str_h=1, str_w=1,
                   bsum=False):
        """
        Create a new ConvLayer parameter object.
        This then is passed as an argument to all the convolution operations.

        N: Number of images in mini-batch
        C: Number of input feature maps
        K: Number of output feature maps

        D: Depth  of input image
        H: Height of input image
        W: Width  of input image

        T: Depth  of filter kernel
        R: Height of filter kernel
        S: Width  of filter kernel

        padding: amount of zero-padding around the given edge
        strides: factor to step the filters by in a given direction

        dtype: need to know dtype to setup proper kernels and params.

        relu: apply a relu to the output for fprop or bprop

        bsum: calculate the sum along the batchnorm axis for fprop or bprop
              outputs an fp32 tensor of size Kx1
        """
        return ConvLayer(self, dtype, N, C, K, D, H, W, T, R, S,
                         pad_d, pad_h, pad_w, str_d, str_h, str_w,
                         bsum)

    def fprop_conv(self, layer, I, F, O, alpha=1.0, beta=0.0, bsum=None, repeat=1):
        assert layer.sizeI == I.size
        assert layer.sizeF == F.size
        assert layer.sizeO == O.size

        layer.fprop_kernels.bind_params(I, F, O, alpha, beta, bsum)

        return self._execute_conv("fprop", layer, layer.fprop_kernels, repeat)

    def bprop_conv(self, layer, gradO, W, gradI, alpha=1.0, beta=0.0, bsum=None, repeat=1):
        assert layer.sizeO == gradO.size
        assert layer.sizeF == W.size
        assert layer.sizeI == gradI.size

        layer.bprop_kernels.bind_params(gradO, W, gradI, alpha, beta, bsum)

        return self._execute_conv("bprop", layer, layer.bprop_kernels, repeat)

    def update_conv(self, layer, I, gradO, gradW, alpha=1.0, repeat=1):
        assert layer.sizeI == I.size
        assert layer.sizeO == gradO.size
        assert layer.sizeF == gradW.size

        layer.updat_kernels.bind_params(I, gradO, gradW, alpha)

        return self._execute_conv("updat", layer, layer.updat_kernels, repeat)

    def _execute_conv(self, op, layer, kernels, repeat):
        # Warmup
        if repeat > 1:
            kernels.execute(max(repeat // 10, 1), unbind=False)

        if self.bench or repeat > 1:
            start, end = _get_events()
            start.record(stream=self.stream)

        kernels.execute(repeat)

#        TODO not sure if this part is needed for cuda kernels?
#        if convert_type:
#            _fp_convert(C_gpudata, convert_type, C, reduce_shape,
#                        self.compute_capability)

        if self.bench or repeat > 1:
            end.record(stream=self.stream)
            end.synchronize()
            msecs  = end.time_since(start) / repeat
            gflops = layer.flops / (msecs * 1000000.0)
            #if layer.TRS[2] == 3:
            print("%7.3f msecs %5.0f gflops %6.0f (%s: %s)" %
                  (msecs, gflops, layer.flops/1000000.0, op, layer))
            return msecs, gflops
        return 0, 0

#@context_dependent_memoize
#def _get_scratch_data(scratch_size):
#    return drv.mem_alloc(scratch_size)

# debugging tool
# import re
# import traceback as tb

# nrv_re = re.compile(r'nervanagpu\.py$')
# def print_trace():
#     caller = None
#     for frame in tb.extract_stack():
#         if GPUTensor.nrv_re.search(frame[0]):
#             break
#         caller = (frame[0],frame[1])
#     print caller
