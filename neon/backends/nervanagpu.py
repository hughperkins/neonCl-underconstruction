# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.
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
import logging
from pycuda.tools import context_dependent_memoize
#from pycuda.curandom import MRG32k3aRandomNumberGenerator as rng_mrg
from pycuda.gpuarray import GPUArray as p_gpuarray
from struct import unpack_from
from pytools import memoize_method
from functools import wraps
from math import log

from neon.backends.backend import Tensor, Backend
from neon.backends.layer_gpu import ConvLayer, _get_sm_count

_none_slice = slice(None, None, None)

logger = logging.getLogger(__name__)


class GPUTensor(Tensor):

    """
    The n-dimensional array data structure that resides in GPU memory,
    and is meant to be manipulated on the GPU.

    Arguments:
        dtype (numpy.ndtype, optional): Underlying data type of the elements.
        allocator (function, optional): Memory allocator.
        base (GPUTensor, optional): The base of the tensor. A tensor can have
                                    different views, this keep tracks of the
                                    original tensor.
        gpudata (pycuda._driver.DeviceAllocation, optional): The actual gpu
                                                             memory that stores
                                                             the tensor.
        strides (tuple, optional): Tuple of bytes to step in each dimension when traversing an
                                   array.
        take_array: The indices of the values to extract.
        is_trans (bool, optional): Whether the tensor is transposed or not.
        rounding (int, optional): Set to desired number of mantissa bits to
                                  stochasicaly round, to set to zero to disable
                                  stochastic rouding.

    See also:
        NervanaGPU class

    Notes:
        Unlike numpy, in this implementation we never collapse dimensions, and
        the minimal number of dimensions will be _min_dims (currently set to 2
        to match cudanet GPU implementation).  So a wrapped scalar will have
        dimension 1x1.
    """

    def __init__(self,
                 backend,
                 shape=None,
                 dtype=np.float32,
                 name=None,
                 persist_values=True,
                 allocator=drv.mem_alloc,
                 base=None,
                 gpudata=None,
                 strides=None,
                 take_array=None,
                 is_trans=False,
                 rounding=0):

        super(GPUTensor, self).__init__(backend, shape, dtype, name,
                                        persist_values)

        # supported dtypes
        assert dtype in (np.float16, np.float32, np.uint8, np.int8, np.uint16,
                         np.int16, np.uint32, np.int32)

        dtype = np.dtype(dtype)

        if isinstance(shape, (tuple, list)) and len(shape) < self._min_dims:
            shape = shape + (1, )

        try:
            size = 1
            for dim in shape:
                size *= dim
        except TypeError:
            assert isinstance(shape, (int, long, np.integer))
            size = shape
            shape = (shape, 1)

        if isinstance(size, np.integer):
            size = np.asscalar(size)

        # only support C ordering for now.
        if strides is None:
            self.strides = _contiguous_strides(shape)
        else:
            self.strides = tuple(strides)

        self.base = base
        self.shape = shape
        self.size = size
        self.dtype = dtype
        self.nbytes = dtype.itemsize * size
        self.allocator = allocator
        self.take_array = take_array
        self.is_trans = is_trans
        self.rounding = rounding
        self.kahan_count = 0
        self.kahan_reset = 0

        if gpudata is None:
            # print "allocate!"
            if size:
                # print(drv.mem_get_info())
                self.gpudata = allocator(self.nbytes)
            else:
                self.gpudata = None

            assert base is None
        else:
            self.gpudata = gpudata

    def __str__(self):
        """
        Returns a string representation of this Tensor.

        Returns:
            str: the representation.
        """
        return ("GPUTensor(base 0x%x) name:%s shape:%s dtype:%s strides:%s "
                "is_trans:%s is_contiguous:%s" % (self.gpudata, self.name,
                                                  self.shape, self.dtype,
                                                  self.strides, self.is_trans,
                                                  self.is_contiguous))

    def __repr__(self):
        """
        Returns a more unambiguous string representation of the Tensor.

        Returns:
            str: The representation.
        """
        return self.__str__()

    def __setitem__(self, index, value):

        self.__getitem__(index)._assign(value)

    def __getitem__(self, index):
        """
        Return a sliced view of an array
        """
        if not isinstance(index, tuple):
            # speed up common case of [:]
            if index == _none_slice:
                return self
            index = (index,)

        new_shape = []
        new_offset = 0
        new_strides = []

        seen_ellipsis = False
        take_array = None

        index_axis = 0
        array_axis = 0

        while index_axis < len(index):

            index_entry = index[index_axis]

            if array_axis > len(self.shape):
                raise IndexError("too many axes in index")

            # Standard slicing (start:stop:step)
            if isinstance(index_entry, slice):
                start, stop, idx_strides = index_entry.indices(
                    self.shape[array_axis])

                array_strides = self.strides[array_axis]

                # def ceil_div(x, y): return -(-x // y)
                new_shape.append(-((start - stop) // idx_strides))
                new_strides.append(idx_strides * array_strides)
                new_offset += array_strides * start * self.dtype.itemsize

                index_axis += 1
                array_axis += 1

            # Fancy indexing
            elif isinstance(index_entry, (GPUTensor, np.ndarray, list, tuple)):

                if isinstance(index_entry, (list, tuple)):
                    index_entry = np.array(index_entry, dtype=np.int32)

                if isinstance(index_entry, np.ndarray):
                    index_entry = self.__class__(
                        self.backend, index_entry.shape, dtype=np.int32).set(index_entry)

                size = max(index_entry.shape)
                if size != index_entry.size:
                    raise IndexError(
                        "Fancy indexing only currently supported dim > 1 in a single dimension.")

                if take_array is not None:
                    raise IndexError(
                        "Fancy indexing only currently supported one axis at a time.")

                if index_entry.dtype.type is not np.int32:
                    # TODO: this should now work for all int types, but need to
                    # test
                    raise IndexError(
                        "Fancy indexing only currently supported with int32 types.")

                take_array = (index_entry, array_axis)

                new_shape.append(size)
                new_strides.append(self.strides[array_axis])

                index_axis += 1
                array_axis += 1

            elif isinstance(index_entry, (int, np.integer)):
                array_shape = self.shape[array_axis]
                if index_entry < 0:
                    index_entry += array_shape

                if not (0 <= index_entry < array_shape):
                    raise IndexError(
                        "subindex in axis %d out of range" % index_axis)

                new_offset += self.strides[array_axis] * \
                    index_entry * self.dtype.itemsize

                if len(self.shape) < 3:
                    new_shape.append(1)
                    new_strides.append(self.strides[array_axis])

                index_axis += 1
                array_axis += 1

            elif index_entry is Ellipsis:
                index_axis += 1

                remaining_index_count = len(index) - index_axis
                new_array_axis = len(self.shape) - remaining_index_count
                if new_array_axis < array_axis:
                    raise IndexError("invalid use of ellipsis in index")
                while array_axis < new_array_axis:
                    new_shape.append(self.shape[array_axis])
                    new_strides.append(self.strides[array_axis])
                    array_axis += 1

                if seen_ellipsis:
                    raise IndexError(
                        "more than one ellipsis not allowed in index")
                seen_ellipsis = True

            else:
                raise IndexError("invalid subindex in axis %d" % index_axis)

        while array_axis < len(self.shape):
            new_shape.append(self.shape[array_axis])
            new_strides.append(self.strides[array_axis])

            array_axis += 1

        return self.__class__(
            backend=self.backend,
            shape=tuple(new_shape),
            dtype=self.dtype,
            allocator=self.allocator,
            base=self,
            gpudata=int(self.gpudata) + new_offset,
            strides=new_strides,
            take_array=take_array,
            name=self.name,
            rounding=self.rounding)

    def _assign(self, value):
        """
        Assign value to the tensor.

        Arguments:
            value (int, float, GPUTensor, OpTreeNode): The value to be assigned.
        """

        stream = self.backend.stream
        if isinstance(value, (int, float)):
            # if we have a contiguous array, then use the speedy driver kernel
            if self.is_contiguous:

                value = self.dtype.type(value)

                if self.dtype.itemsize == 1:
                    drv.memset_d8_async(
                        self.gpudata, unpack_from('B', value)[0], self.size, stream)
                elif self.dtype.itemsize == 2:
                    drv.memset_d16_async(
                        self.gpudata, unpack_from('H', value)[0], self.size, stream)
                else:
                    drv.memset_d32_async(
                        self.gpudata, unpack_from('I', value)[0], self.size, stream)

            # otherwise use our copy kerel
            else:
                OpTreeNode.build("assign", self, value)

        elif isinstance(value, GPUTensor):
            # TODO: add an is_binary_compat like function
            if self.is_contiguous and value.is_contiguous and self.dtype == value.dtype:
                drv.memcpy_dtod_async(
                    self.gpudata, value.gpudata, self.nbytes, stream)
            else:
                OpTreeNode.build("assign", self, value)

        # collapse and execute an op tree as a kernel
#        elif isinstance(value, OpTreeNode):
#            OpTreeNode.build("assign", self, value)

        # assign to numpy array (same as set())
        elif isinstance(value, np.ndarray):
            self.set(value)

        else:
            raise TypeError("Invalid type for assignment: %s" % type(value))

        return self

    def set(self, ary):
        """
        Copy host array to device.

        Arguments:
            ary: host array, needs to be contiguous

        Returns:
            GPUTensor: self
        """
        stream = self.backend.stream
        assert ary.size == self.size
        assert self.is_contiguous, "Array in set() must be contiguous"
        if ary.dtype is not self.dtype:
            ary = ary.astype(self.dtype)
        if ary.ndim < self._min_dims:
            ary = ary.reshape(ary.size, 1)
        assert ary.strides == tuple(
            self.dtype.itemsize * s for s in self.strides)

        drv.memcpy_htod_async(self.gpudata, ary, stream)

        return self

    @property
    @memoize_method
    def is_contiguous(self):
        """
        Returns whether the memory of the tensor is contiguous.

        Return
            bool: Whether the memory of the tensor is contiguous.
        """
        return not self.take_array and self.strides == _contiguous_strides(self.shape)


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

    # size of the RNG pool on device
    # currently this is hard wired
    def __init__(self,
#                 rng_seed=None,
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

        # store the rand pool for each context
        self.context_rand_state_map = {}  # stores gpu memory reference
        self.context_rand_state_alive = {}  # set whether randstate is fresh

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
        self.hist_base = drv.mem_alloc(self.hist_bins * self.hist_max * 4)
        drv.memset_d32(self.hist_base, 0, self.hist_bins * self.hist_max)

        self.compute_capability = (4,0)
        self.use_cudac_kernels = True

        self.enable_winograd = enable_winograd
        self.cache_dir = cache_dir
        if not os.path.isdir(self.cache_dir):
            os.makedirs(self.cache_dir)

    def set_scratch_size(self, *args):

        total_size = 0
        for size in args:
            if size & 127 != 0:
                size += 128 - (size & 127)
            total_size += size

        if total_size > self.scratch_size:
            self.scratch_size = total_size

    def __del__(self):
        try:
            self.ctx.detach()
        except:
            pass

    def execute(self, optree):
        """
        Execute the optree. Break optree into sub-optrees if necessary.
        """
        raise Exception('not implemented')

    def empty(self, shape, dtype=None, name=None, persist_values=True,
              parallel=False, distributed=False, allocator=drv.mem_alloc):
        """
        Allocate the space for a GPUTensor
        """
        dtype = self.default_dtype if dtype is None else dtype
        return GPUTensor(self, shape, dtype=dtype, name=name,
                         persist_values=persist_values, allocator=allocator,
                         rounding=self.round_mode)

    def conv_layer(self, dtype,
                   N, C, K,
                   D=1, H=1, W=1,
                   T=1, R=1, S=1,
                   pad_d=0, pad_h=0, pad_w=0,
                   str_d=1, str_h=1, str_w=1,
                   relu=False, bsum=False):
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
                         relu, bsum)

    def fprop_conv(self, layer, I, F, O, alpha=1.0, beta=0.0, bsum=None, repeat=1):
        assert layer.sizeI == I.size
        assert layer.sizeF == F.size
        assert layer.sizeO == O.size

        layer.fprop_kernels.bind_params(I, F, O, alpha, beta, bsum)

        return self._execute_conv("fprop", layer, layer.fprop_kernels, repeat)

    def bprop_conv(self, layer, F, E, grad_I, alpha=1.0, beta=0.0, bsum=None, repeat=1):
        assert layer.sizeF == F.size
        assert layer.sizeO == E.size
        assert layer.sizeI == grad_I.size

        layer.bprop_kernels.bind_params(E, F, grad_I, alpha, beta, bsum)

        return self._execute_conv("bprop", layer, layer.bprop_kernels, repeat)

    def update_conv(self, layer, I, E, grad_F, alpha=1.0, repeat=1):
        assert layer.sizeI == I.size
        assert layer.sizeO == E.size
        assert layer.sizeF == grad_F.size

        layer.updat_kernels.bind_params(I, E, grad_F, alpha)

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

# Note the strides computed here do not include the dtype.itemsize
def _contiguous_strides(shape):
    if shape:
        strides = [1]
        for s in shape[:0:-1]:
            strides.append(strides[-1] * s)
        return tuple(strides[::-1])
    else:
        return ()


@context_dependent_memoize
def _get_scratch_data(scratch_size):
    return drv.mem_alloc(scratch_size)


@context_dependent_memoize
def _get_events():
    return (drv.Event(), drv.Event())

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
