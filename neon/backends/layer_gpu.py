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
#from neon.backends import kernel_specs
from neon.backends import convolution
from pycuda.tools import context_dependent_memoize
from operator import mul
from math import ceil
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
        self.weights   = None
        self.fprop_in  = None
        self.fprop_out = None
        self.bprop_in  = None
        self.bprop_out = None

        self.learning_rate = 0.0

    def init_activations(self, fprop_out=None):

        if fprop_out is not None:
            self.fprop_out = fprop_out
        else:
            self.fprop_out = self.lib.empty(self.dimO, dtype=self.dtype)

        self.act_stats = self.lib.empty((self.dimO2[0], 1), dtype=np.float32)

    def init_deltas(self, shared=None):

        if shared is None:
            self.bprop_out = self.lib.empty(self.dimI, dtype=self.dtype)
        else:
            self.bprop_out = shared[0].share(self.dimI)
            shared.reverse()

        self.delta_stats = self.lib.empty((self.dimI2[0], 1), dtype=np.float32)

    def init_weights(self, loc=0.0, scale=0.1, shared=None, zeros=False):

        if self.sizeF > 0:
            if zeros:
                self.weights  = self.lib.zeros(self.dimF, dtype=self.dtype)
            else:
                weights       = np.random.normal(loc, scale, self.dimF)
                self.weights  = self.lib.array(weights, dtype=self.dtype)

            if shared is None:
                self.updat_out = self.lib.empty(self.dimF, dtype=self.dtypeU)
            else:
                self.updat_out = shared.share(self.dimF, dtype=self.dtypeU)

            self.weight_stats = self.lib.empty((self.dimF2[0], 1), dtype=np.float32)

    def scale_weights(self, scale):

        mean = self.get_activation_mean()
        self.weights[:] *= scale/mean

    def fprop(self, fprop_in, scale_weights=0):
        if self.fprop_in is None and fprop_in:
            self.fprop_in = fprop_in.reshape(self.dimI)
        return self.fprop_in

    def bprop(self, bprop_in, beta=0):
        return bprop_in

    def grad_descent(self):
        self.weights[:] += self.updat_out*self.learning_rate

    def fprop_stats(self):
        print("fprop:%10.5f mean %11.5f max %s"
              % (self.get_activation_mean(), self.get_activation_max(), self))

    def bprop_stats(self):
        if self.bprop_out is not None:
            print("bprop:%10.5f mean %11.5f max %s"
                  % (self.get_delta_mean(), self.get_delta_max(), self))

        if self.weights is not None:
            up_mean, up_max = (self.get_update_mean(), self.get_update_max())
            wt_mean, wt_max = (self.get_weight_mean(), self.get_weight_max())
            rt_mean, rt_max = (0.0001 * up_mean/wt_mean, 0.0001 * up_max/wt_max)
            print("updat:%10.5f mean %11.5f max %s" % (up_mean, up_max, self))
            print("weigh:%10.5f mean %11.5f max" % (wt_mean, wt_max))
            print("ratio:%10.5f mean %11.5f max" % (rt_mean, rt_max))

    @staticmethod
    def create(lib, conf, prev_layer, dtype):

        config     = dict(conf)
        layer_type = config.pop("layer")

        # merge dtype specific settings
        config["dtype"] = dtype

        # merge shared params
        config.update(config.pop("common", {}))

        # Propagate the fixed and calculated dimensions
        if prev_layer is not None:
            config["N"] = prev_layer.N

            if layer_type is FullLayer:
                config["nIn"] = prev_layer.nOut
            elif layer_type is PoolLayer and type(prev_layer) is FullLayer:
                config["C"] = prev_layer.nOut
            elif layer_type is BatchNorm and type(prev_layer) is FullLayer:
                config["nIn"] = prev_layer.nOut
            else:
                config["C"] = prev_layer.K
                config["D"] = prev_layer.M
                config["H"] = prev_layer.P
                config["W"] = prev_layer.Q

                if layer_type is Inception:
                    partitions  = config.pop("partitions")
                    config["K"] = 0

                    config["partitions"] = []
                    for part in partitions:
                        layer_sequence = []
                        part_prev_layer = prev_layer
                        for layer_conf in part:
                            part_prev_layer = Layer.create(lib, layer_conf, part_prev_layer, dtype)
                            layer_sequence.append(part_prev_layer)

                        last = layer_sequence[-1]
                        config["partitions"].append(layer_sequence)
                        config["K"] += last.K
                        if "P" in config:
                            assert config["P"] == last.P and config["Q"] == last.Q
                        else:
                            config["M"] = last.M
                            config["P"] = last.P
                            config["Q"] = last.Q

        # Instantiate the layer
        return layer_type(lib, **config)



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
                 relu=False, bsum=False):

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
        self.relu = relu
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

        ####### Cuda C ###########
        if lib.use_cudac_kernels:
            print('cudac')
            #3D conv not supported yet
            if T > 1 or D > 1:
                raise ValueError("3D Convolution not supported by CUDA C kernels.")

            if relu:
                raise ValueError("Compound relu not supported by CUDA C kernels.")

            self.fprop_kernels = convolution.FpropCuda(lib, self.dtype, N, C, K, D, H, W, T, R, S, M, P, Q,
                                                       pad_d, pad_h, pad_w, str_d, str_h, str_w, bsum=bsum)
            # TODO small C bprop?
            self.bprop_kernels = convolution.BpropCuda(lib, self.dtype, N, C, K, D, H, W, T, R, S, M, P, Q,
                                                       pad_d, pad_h, pad_w, str_d, str_h, str_w, bsum=bsum)
            self.updat_kernels = convolution.UpdateCuda(lib, self.dtype, N, C, K, D, H, W, T, R, S, M, P, Q,
                                                        pad_d, pad_h, pad_w, str_d, str_h, str_w)

        logger.debug("%s: %s, %s, %s", str(self), str(self.fprop_kernels), str(self.bprop_kernels), str(self.updat_kernels))


    def init_activations(self, fprop_out=None):

        super(ConvLayer, self).init_activations(fprop_out)

        if self.bsum:
            self.batch_sum = self.lib.empty(self.dimS, dtype=np.float32)
        else:
            self.batch_sum = None

    def fprop(self, fprop_in, scale_weights=0):
        """
        Conv Layer forward propagation.

        Arguments:
            fprop_in (Tensor): Inputs
            scale_weights (float): Scale weights by scale/mean if nonzero

        Returns:
            fprop_out (Tensor): Output activations
        or
            (self.fprop_out, self.batch_sum) (tuple): Tuple with batch_sum
                added as the second entry.
        """
        fprop_in = super(ConvLayer, self).fprop(fprop_in)
        self.lib.fprop_conv(self, fprop_in, self.weights, self.fprop_out, bsum=self.batch_sum)

        if scale_weights:
            self.scale_weights(scale_weights)
            self.fprop(fprop_in)

        if self.bsum:
            return (self.fprop_out, self.batch_sum)
        return self.fprop_out

    def bprop(self, bprop_in, beta=0):

        if self.relu:
            self.bprop_relu(bprop_in)
        if self.bprop_out is not None:
            self.lib.bprop_conv(self, self.weights, bprop_in, self.bprop_out, beta=beta)

        self.lib.update_conv(self, self.fprop_in, bprop_in, self.updat_out)
        self.grad_descent()

        return self.bprop_out

    def __str__(self):
        return ("ConvLayer: NCK: (%3d, %3d, %3d) HW:%s" %
                (self.N, self.C, self.K, self.DHW[1:3]))



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
