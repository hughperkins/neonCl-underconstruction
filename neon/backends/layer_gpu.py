# Copyright 2014 Nervana Systems Inc. All rights reserved.
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
Definition of the GPU layers
These layers are mainly used for old benchmarking code,
but they also cache all the computed params for complex layers.
TODO: clean up merge with CPU layers
TODO: remove any non-param caching code, neon layers should replace benchmark code.
"""
import logging
import numpy as np
import pycuda.driver as drv
from neon.backends import convolution
from pycuda.tools import context_dependent_memoize
from operator import mul
import sys


logger = logging.getLogger(__name__)


if sys.version_info >= (3, 0):
    from functools import reduce


class Layer(object):

    """
    GPU Layer base class
    """

    def __init__(self, lib, dtype, N, dtypeU=None):

        if hasattr(dtype, 'type'):
            self.dtype = dtype
        else:
            self.dtype = np.dtype(dtype)

        self.N      = N
        self.dtypeU = dtype if dtypeU is None else dtypeU
        self.lib    = lib
        self.flops  = 0
        self.sizeI  = 0
        self.sizeO  = 0
        self.sizeF  = 0

        self.learning_rate = 0.0


class ConvLayer(Layer):

    """
    ConvLayer parameter object.
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
    """

    def __init__(self, lib, dtype,
                 N, C, K,
                 D=1, H=1, W=1,
                 T=1, R=1, S=1,
                 pad_d=0, pad_h=0, pad_w=0,
                 str_d=1, str_h=1, str_w=1,
                 bsum=False):

        super(ConvLayer, self).__init__(lib, dtype, N, np.float32)

        # Compute the output spatial dimensions
        M = lib.output_dim(D, T, pad_d, str_d)
        P = lib.output_dim(H, R, pad_h, str_h)
        Q = lib.output_dim(W, S, pad_w, str_w)

        self.C = C
        self.K = K
        self.M = M
        self.P = P
        self.Q = Q
        self.NCK = (N, C, K)
        self.TRS = (T, R, S)
        self.DHW = (D, H, W)
        self.MPQ = (M, P, Q)
        self.padding = (pad_d, pad_h, pad_w)
        self.strides = (str_d, str_h, str_w)
        self.bsum = bsum

        self.all_params = (N, C, K, D, H, W, T, R, S, pad_d, pad_h, pad_w, str_d, str_h, str_w)

        self.dimI   = (C, D, H, W, N)
        self.dimF   = (C, T, R, S, K)
        self.dimFb  = (K, T, R, S, C)
        self.dimO   = (K, M, P, Q, N)
        self.dimI2  = (C*D*H*W, N)
        self.dimF2  = (C*T*R*S, K)
        self.dimF2t = (K, C*T*R*S)
        self.dimO2  = (K*M*P*Q, N)
        self.dimS   = (K, 1)
        self.sizeI  = reduce(mul, self.dimI, 1)
        self.sizeF  = reduce(mul, self.dimF, 1)
        self.sizeO  = reduce(mul, self.dimO, 1)
        self.nOut   = reduce(mul, self.MPQ, 1) * K

        # flop count for benchmarking
        self.flops = P*Q*M*K*N*C*R*S*T * 2.0

        if T > 1 or D > 1:
            raise ValueError("3D Convolution not supported by CUDA C kernels.")

        self.fprop_kernels = convolution.FpropCuda(lib, self.dtype, N, C, K, D, H, W, T, R, S, M, P, Q,
                                                   pad_d, pad_h, pad_w, str_d, str_h, str_w, bsum=bsum)
        # TODO small C bprop?
        self.bprop_kernels = convolution.BpropCuda(lib, self.dtype, N, C, K, D, H, W, T, R, S, M, P, Q,
                                                   pad_d, pad_h, pad_w, str_d, str_h, str_w, bsum=bsum)
        self.updat_kernels = convolution.UpdateCuda(lib, self.dtype, N, C, K, D, H, W, T, R, S, M, P, Q,
                                                    pad_d, pad_h, pad_w, str_d, str_h, str_w)

#        logger.debug("%s: %s, %s, %s", str(self), str(self.fprop_kernels), str(self.bprop_kernels), str(self.updat_kernels))


# Magic numbers and shift amounts for integer division
# Suitable for when nmax*magic fits in 32 bits
# Shamelessly pulled directly from:
# http://www.hackersdelight.org/hdcodetxt/magicgu.py.txt
def _magic32(nmax, d):
    nc = ((nmax + 1) // d) * d - 1
    nbits = len(bin(nmax)) - 2
    for p in range(0, 2 * nbits + 1):
        if 2 ** p > nc * (d - 1 - (2 ** p - 1) % d):
            m = (2 ** p + d - 1 - (2 ** p - 1) % d) // d
            return (m, p)
    raise ValueError("Can't find magic number for division")

# Magic numbers and shift amounts for integer division
# Suitable for when nmax*magic fits in 64 bits and the shift
# lops off the lower 32 bits
def _magic64(d):
    # 3 is a special case that only ends up in the high bits
    # if the nmax is 0xffffffff
    # we can't use 0xffffffff for all cases as some return a 33 bit
    # magic number
    nmax = 0xffffffff if d == 3 else 0x7fffffff
    magic, shift = _magic32(nmax, d)
    if magic != 1:
        shift -= 32
    return (magic, shift)

# flatten a nested list of lists or values
def _flatten(lst):
    return sum(([x] if not isinstance(x, (list, tuple))
                else _flatten(x) for x in lst), [])

def _ceil_div(x, y):
    return -(-x // y)

@context_dependent_memoize
def _get_sm_count():
    attributes = drv.Context.get_device().get_attributes()
    return attributes[drv.device_attribute.MULTIPROCESSOR_COUNT]
