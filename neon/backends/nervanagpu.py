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
from pycuda.curandom import MRG32k3aRandomNumberGenerator as rng_mrg
from pycuda.gpuarray import GPUArray as p_gpuarray
from struct import unpack_from
from pytools import memoize_method
from functools import wraps
from math import log

from neon.backends import kernel_specs
from neon.backends.backend import Tensor, Backend
from neon.backends.layer_gpu import ConvLayer, _get_sm_count
#from neon.backends.kernels.cuda import pooling, roipooling
from scikits.cuda import cublas

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

    def __len__(self):
        """
        Returns the size of the leading dimension of self.

        Returns:
            int: The size of the leading dimension.
        """
        if len(self.shape):
            return self.shape[0]
        else:
            return 0

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

    def __int__(self):
        """
        Returns an integer representation of the underlying gpu memory buffer.

        Returns:
            int: The int representation
        """
        return int(self.gpudata)

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

    def get(self, stream=None):
        """
        Copy device array to host.

        Returns:
            numpy.ndarray: A host numpy array
        """

        if self.is_contiguous:
            ary = np.empty(self.shape, self.dtype)
            drv.memcpy_dtoh_async(ary, self.gpudata, stream)
        else:
            # if it is not contiguous, need to copy it over to new device mem
            ary_d = self.backend.empty(self.shape, self.dtype)
            ary_d.copy(self)
            ary = np.empty(self.shape, self.dtype)
            drv.memcpy_dtoh_async(ary, ary_d.gpudata, stream)
        return ary

    def raw(self):
        """
        Access the raw buffer.

        Returns:
            pointer: A device specific pointer
        """
        return self.gpudata

    def asnumpyarray(self):
        """
        Deprecated.
        Scheduled to be removed in 2.0.
        Use get() instead.
        """
        return self.get()

    def asbuffer(self):
        """
        asbuffer returns buffer interface to gpu data
        """
        return self.gpudata.as_buffer(self.nbytes)

    def take(self, indices, axis, out=None):
        """
        Take elements from an array along an axis.
        """
        if axis == 1:
            view = self.__getitem__((_none_slice, indices))
        else:
            view = self.__getitem__((indices, _none_slice))

        if out:
            return out._assign(view)
        return view

    def fill(self, value):
        return self._assign(value)

    def copy(self, a):
        return self._assign(a)

    def copy_from(self, a):
        """ alias of copy"""
        return self.set(a)

    def reshape(self, *shape):
        """
        return a reshaped view
        """
        if isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])

        if len(shape) < self._min_dims:
            shape = shape + (1, )

        if -1 in shape:
            missing_dim = -self.size / np.prod(shape)
            shape = tuple([missing_dim if x == -1 else x for x in shape])

        if shape == self.shape:
            return self

        size = np.prod(shape)

        if size != self.size:
            raise ValueError("total size of new array must be unchanged")

        if not self.is_contiguous:
            raise TypeError("reshaping of non-contiguous arrays is not yet supported")

        return self.__class__(
            backend=self.backend,
            shape=shape,
            dtype=self.dtype,
            allocator=self.allocator,
            base=self,
            gpudata=self.gpudata,
            strides=_contiguous_strides(shape),
            name=self.name,
            rounding=self.rounding)

    @property
    def T(self):
        """
        return a transposed view
        """
        if len(self.shape) <= 2:
            shape = self.shape[::-1]
            strides = self.strides[::-1]
        else:
            # support for batched dot.
            # perserve outer dimension but reverse inner dims
            shape = list(self.shape[::-1])
            strides = list(self.strides[::-1])
            shape = tuple(shape[-1:] + shape[:-1])
            strides = tuple(strides[-1:] + strides[:-1])

        return self.__class__(
            backend=self.backend,
            shape=shape,
            dtype=self.dtype,
            allocator=self.allocator,
            base=self,
            gpudata=self.gpudata,
            strides=strides,
            is_trans=not self.is_trans,
            name=self.name,
            rounding=self.rounding)

    def transpose(self, out=None):
        """
        Return a transposed view of the data.  Alias of .T property needed for
        MOP compatibility.
        """
        if out:
            return OpTreeNode.build("assign", out, self.T)
        return self.T

    def share(self, shape, dtype=None, name=None):
        """
        return a view: ary, where ary.size <= self.size
        Allows easy sharing of temporary memory
        """
        size = np.prod(shape)
        if size > self.size:
            raise ValueError("total size of new array must <= size of parent")

        if not self.is_contiguous:
            raise TypeError("sharing of non-contigous "
                            "arrays is not yet supported")

        if dtype is None:
            dtype = self.dtype
        else:
            dtype = np.dtype(dtype)

        new_base = self if self.base is None else self.base

        return self.__class__(
            backend=self.backend,
            shape=shape,
            dtype=dtype,
            allocator=self.allocator,
            base=new_base,
            gpudata=self.gpudata,
            strides=_contiguous_strides(shape),
            name=name,
            rounding=self.rounding)

    def hist(self, tag):
        """
        Compute a histogram of the current tensor values.

        Arguments:
            tag (string): Tag to identify the current state of the tensor,
                          useful for disambiguating multiple histograms of the
                          same tensor at different points in time.

        Returns:
            Tensor containing the histogram data.

        """
        nbins = self.backend.hist_bins
        offset = self.backend.hist_offset
        from neon.backends.float_ew import _compute_hist
        hist_tensor = self.backend._hist_tensor(tag)
        _compute_hist(self, hist_tensor.gpudata, nbins, offset)
        return hist_tensor

    @property
    def ptr(self):
        """
        Returns an integer representation of the underlying gpu memory buffer.

        Returns:
            int: The int representation
        """
        return self.gpudata.__int__()

    @property
    @memoize_method
    def is_contiguous(self):
        """
        Returns whether the memory of the tensor is contiguous.

        Return
            bool: Whether the memory of the tensor is contiguous.
        """
        return not self.take_array and self.strides == _contiguous_strides(self.shape)


def memoize_stacks(func):
    """
    memoize the stacks using intrinsic_key_maps
    """
    cache = {}

    @wraps(func)
    def memoizer(be, optree):
        optree_key, tensor_index_map, index_tensor_map = optree.intrinsic_key_maps()
        # make sure it's the same backend
        optree_key = (optree_key, id(be))
        if optree_key in cache:
            # replace tensors
            stacks, cached_tensor_index_map = cache[optree_key]
            for stack in stacks:
                for i in range(len(stack)):
                    if isinstance(stack[i], Tensor):
                        if stack[i] in cached_tensor_index_map:
                            stack[i] = index_tensor_map[
                                cached_tensor_index_map[stack[i]]]
            # update the cached_tensor_index_map
            cache[optree_key] = (stacks, tensor_index_map)
        else:
            # cache stacks and tensor_index_map
            # print ('created memoize stack')
            stacks = func(be, optree)
            cache[optree_key] = (stacks, tensor_index_map)
        return stacks

    return memoizer


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
    _RNG_POOL_SIZE = (3*2048*32, 1)
    def __init__(self,
                 rng_seed=None,
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
        super(NervanaGPU, self).__init__(rng_seed,
                                         default_dtype,
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

        # Fall back to CUDA C kernels on older (pre-Maxwell) GPU generations
        self.compute_capability = drv.Device(self.device_id).compute_capability()
        if self.compute_capability[0] < 5:
            self.use_cudac_kernels = True
            self.cublas_handle = cublas.cublasCreate()

            logger.warn("Neon is highly optimized for Maxwell GPUs. Although "
                        "you might get speedups over CPUs, note that you are "
                        "running on a pre-Maxwell GPU and you might not "
                        "experience the fastest performance. For faster "
                        "performance using the Nervana Cloud contact "
                        "info@nervanasys.com")
        else:
            self.use_cudac_kernels = False
        self.compute_capability = (4,0)
        self.use_cudac_kernels = True
        self.cublas_handle = cublas.cublasCreate()

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

        return int(_get_scratch_data(self.scratch_size))

    def scratch_buffer_offset(self, size):

        if size & 127 != 0:
            size += 128 - (size & 127)

        if size + self.scratch_offset > self.scratch_size:
            raise RuntimeError("nervanagpu.scratch_size(%d) is too small for this operation." % self.scratch_size)

        data = int(_get_scratch_data(self.scratch_size)) + self.scratch_offset
        self.scratch_offset += size

        return data

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

    def get_events(self):
        return _get_events()

    def gen_rng(self, seed=None):
        """
        Generate the random number generator on device and on host

        Arguments:
            seed (int): random number generator seed

        Returns:
            seeded numpy RNG
        """
        # generate on host rng
        self.rng = np.random.RandomState(seed)

        # this RNG is for handling normally distributed numbers on device
        self.pcg = rng_mrg()
        # save the initial state of host rng
        self.init_rng_state = self.rng.get_state()

        # generate random integers to seed the LSFR
        # RNGs on the device
        self.init_rng_state_dev = self._gen_dev_randstate()

        # call below is mainly to set on device RNG states
        # to self.init_rng_state_dev
        self.rng_reset()

        # if the current context already has an rng clear it
        ctx = drv.Context.get_current()
        if ctx in self.context_rand_state_alive:
            self.context_rand_state_alive[ctx] = False

        # generate the on device RNG
        self._set_rand_state_dev(state=self.init_rng_state_dev)
        return self.rng

    def _gen_dev_randstate(self):
        """
        Generate a list of random uint32 numbers to seed the LFSR
        states on device

        Returns:
            np.array: return a vector of uint32 numbers
        """
        # will use the numpy rng to generate the states
        # but want to reset it after this is done
        state_save = self.rng.get_state()

        # smaller number for 32bit systems
        maxexp = 32 if sys.maxint > 2**32 else 30

        # draw _RNG_POOL_SIZE 32 bit ints to seed LFSR on device
        # lower bound 1 to avoid seeding LFSR with 0
        rand_init = self.rng.random_integers(1, 2**maxexp - 1, NervanaGPU._RNG_POOL_SIZE)
        rand_init = rand_init.astype(np.uint32)

        # put the numpy (on host) RNG back to its state before
        self.rng.set_state(state_save)

        return rand_init

    def rng_reset(self):
        """
        Reset the RNG to the initial state stored in
        self.init_rng_state and self.init_rng_state_dev
        for the host and device RNG, respectively.
        """
        self.rng_set_state( (self.init_rng_state, self.init_rng_state_dev) )

    def rng_set_state(self, rng_states):
        """
        Set the RNG state for both the on device and on host RNGs

        Arguments:
            rng_states (tuple of np.arrays): tuple with 2 elements
                                                1) numpy random number state vector
                                                2) array of uint32 specifying on dev RNG state
        """
        assert type(rng_states) is tuple and len(rng_states) == 2
        self._set_rand_state_dev(state=rng_states[1])
        self.rng.set_state(rng_states[0])

    def rng_get_state(self):
        """
        Return the current state of the on-host and on-device RNGs

        Returns:
            (np.array, np.array): the on-host and on-device RNG state vectors,
                                  respectively
        """
        dev_state = self._get_rand_state_dev()
        dev_state_local = np.zeros(NervanaGPU._RNG_POOL_SIZE).astype(np.uint32)
        drv.memcpy_dtoh(dev_state_local, dev_state)
        return (self.rng.get_state(), dev_state_local)

    def _set_rand_state_dev(self, state=None):
        """
        Set on device RNG states to values given by "state" input.

        Arguments:
            state (np.array or None): an array of uint32 values used to
                                      set the state of the on device LFSRs.
                                      if set to None, the state will be created
                                      randomly
        """
        ctx = drv.Context.get_current()
        if state is None:
            state = self._gen_dev_randstate()
        if ctx in self.context_rand_state_map:
            rand_state = self.context_rand_state_map[ctx]
        else:
            rand_state = drv.mem_alloc(state.nbytes)
            self.context_rand_state_map[ctx] = rand_state
        drv.memcpy_htod(rand_state, state)
        self.context_rand_state_alive[ctx] = True
        return

    def fill_normal(self, ary, mean=0, stdv=1):
        """
        Fills ary with gaussian noise with given mean and std dev.
        """
        self.pcg.fill_normal(p_gpuarray(ary.shape, ary.dtype, gpudata=ary.gpudata))
        if not all([mean==0, stdv==1]):
            ary[:] = ary * stdv + mean

    @memoize_stacks
    def _split_to_stacks(self, optree):
        """
        split an optree to stacks
        """
        # post-order traversal
        whole_stack = optree.traverse(list())

        # build stages, each stage contains a sub optree
        stages = []
        main_stage = []
        main_stage_axis = []

        # get minority axis for binary operation default, suports axis 0 and 1
        axis_count = [0, 0]
        for s in whole_stack:
            if isinstance(s, dict) and s['op'] in OpCollection.reduction_ops:
                assert s['axis'] == 0 or s['axis'] == 1
                axis_count[s['axis']] += 1
        minority_axis = 0 if axis_count[0] <= axis_count[1] else 1

        # traverse stack and split stages
        for s in whole_stack:
            if isinstance(s, dict):
                if s['op'] == 'dot':
                    # convert left and right child to tensor when it was not
                    right = main_stage.pop()
                    main_stage_axis.pop()  # don't care the value
                    left = main_stage.pop()
                    main_stage_axis.pop()  # don't care the value
                    if isinstance(left, OpTreeNode):
                        left_buf = self._buf_malloc(left.shape)
                        stages.append(OpTreeNode({"op": "assign"}, left_buf,
                                                 left))
                        left = left_buf
                    if isinstance(right, OpTreeNode):
                        right_buf = self._buf_malloc(right.shape)
                        stages.append(OpTreeNode({"op": "assign"}, right_buf,
                                                 right))
                        right = right_buf
                    # buffer to store the result of dot
                    buf = self._buf_malloc((left.shape[0], right.shape[1]))
                    # save to stages
                    stages.append(OpTreeNode({"op": "assign"}, buf,
                                             OpTreeNode(s, left, right)))
                    # push buf to main_stage
                    main_stage.append(buf)
                    main_stage_axis.append(None)
                elif s['op'] == 'transpose':
                    # the object being transposed must be optree here
                    operand = main_stage.pop()
                    main_stage_axis.pop()  # don't care the value
                    # allocate buf for the operand shape
                    buf = self._buf_malloc(operand.shape)
                    # evaluate to buf
                    stages.append(OpTreeNode({"op": "assign"}, buf, operand))
                    # put the buf back to main_stage
                    main_stage.append(buf.T)
                    main_stage_axis.append(None)
                elif s['op'] in OpCollection.reduction_ops:
                    # since 2d reduction is converted
                    assert s['axis'] is not None
                    operand = main_stage.pop()
                    prev_axis = main_stage_axis.pop()
                    if prev_axis is not None and prev_axis != s['axis']:
                        # put everything under previous reduction to buf
                        buf = self._buf_malloc(operand.shape)
                        stages.append(
                            OpTreeNode({"op": "assign"}, buf, operand))
                        # put the buf with current reduction to main stage
                        main_stage.append(OpTreeNode(s, buf, None))
                        main_stage_axis.append(s['axis'])
                    else:
                        # do standary OpCollection.unary_ops
                        main_stage.append(OpTreeNode(s, operand, None))
                        main_stage_axis.append(s['axis'])
                elif s['op'] in OpCollection.unary_ops:
                    # will not run into multiple-axis reduction problem
                    # just pop, build optree and put back
                    operand = main_stage.pop()
                    axis = main_stage_axis.pop()
                    main_stage.append(OpTreeNode(s, operand, None))
                    main_stage_axis.append(axis)  # cancelled out
                elif s['op'] in OpCollection.binary_ops:  # not dot
                    # binary ops might run into multiple-axis reduction
                    right = main_stage.pop()
                    prev_axis_right = main_stage_axis.pop()
                    left = main_stage.pop()
                    prev_axis_left = main_stage_axis.pop()
                    if (prev_axis_right is not None and
                            prev_axis_left is not None and
                            prev_axis_left != prev_axis_right):
                        # do reduction on minority axis
                        if prev_axis_left == minority_axis:
                            buf = self._buf_malloc(left.shape)
                            stages.append(
                                OpTreeNode({"op": "assign"}, buf, left))
                            left = buf
                            axis = prev_axis_right
                        else:
                            buf = self._buf_malloc(right.shape)
                            stages.append(
                                OpTreeNode({"op": "assign"}, buf, right))
                            right = buf
                            axis = prev_axis_left
                        # append to main stage
                        main_stage.append(OpTreeNode(s, left, right))
                        main_stage_axis.append(axis)
                    else:
                        # no multiple-axis reduction, perform standard process
                        main_stage.append(OpTreeNode(s, left, right))
                        axis = None
                        if prev_axis_left is not None:
                            axis = prev_axis_left
                        else:
                            axis = prev_axis_right
                        main_stage_axis.append(axis)
                else:
                    return NotImplemented
            else:
                # tensor or scalars, just push to main_stage
                main_stage.append(s)
                main_stage_axis.append(None)

        # append the the laste stage
        stages.append(main_stage[0])

        # build stacks for call_compound_kernel
        stacks = []
        for stage in stages:
            # now all stages is exact one simple optree
            assert(isinstance(stage, OpTreeNode))
            # create stack
            stacks.append(stage.traverse(list()))

        # free buffer from buf_active to buf, without loosing the reference
        self._buf_free()

        return stacks

    def _is_simple_stack(self, stack):
        """
        TODO move this to _split_to_stacks, deal with memoize better
        TODO add test to this func
        """
        reduction_axes = [False, False]
        for s in stack:
            if isinstance(s, dict):
                if s['op'] == 'dot' or s['op'] == 'transpose':
                    return False
                elif s['op'] in OpCollection.reduction_ops:
                    reduction_axes[s['axis']] = True
                    if reduction_axes[1 - s['axis']]:
                        return False
        return True

    def execute(self, optree):
        """
        Execute the optree. Break optree into sub-optrees if necessary.
        """
        from neon.backends.float_ew import call_compound_kernel

        # get post order stack
        stack = optree.traverse(list())

        # bypass stage creation
        if self._is_simple_stack(stack):
            return call_compound_kernel(self._get_rand_state_dev(), self.compute_capability, *stack)

        # create stages and evaluate
        stacks = self._split_to_stacks(optree)

        for stack in stacks:
            if (len(stack) == 5 and isinstance(stack[3], dict) and
                    stack[3]['op'] == 'dot'):
                # evaluate the simple dot
                self.compound_dot(stack[1], stack[2], stack[0])
            else:
                call_compound_kernel(self._get_rand_state_dev(), self.compute_capability, *stack)

        return stacks[-1][0]  # TODO: to be removed, used in partial

    def empty(self, shape, dtype=None, name=None, persist_values=True,
              parallel=False, distributed=False, allocator=drv.mem_alloc):
        """
        Allocate the space for a GPUTensor
        """
        dtype = self.default_dtype if dtype is None else dtype
        return GPUTensor(self, shape, dtype=dtype, name=name,
                         persist_values=persist_values, allocator=allocator,
                         rounding=self.round_mode)

    def array(self, ary, dtype=None, name=None, persist_values=True,
              parallel=False, distributed=False, allocator=drv.mem_alloc):
        """
        converts a numpy array to a GPUTensor
        """
        dtype = self.default_dtype if dtype is None else dtype
        if ary.ndim < self._min_dims:
            ary = ary.reshape(ary.size, 1)
        return GPUTensor(self, ary.shape, dtype=dtype, name=name,
                         persist_values=persist_values, allocator=allocator,
                         rounding=self.round_mode).set(ary)

    def zeros(self, shape, dtype=None, name=None, persist_values=True,
              parallel=False, distributed=False, allocator=drv.mem_alloc):
        """
        Returns an array of the given shape and dtype filled with 0's.
        """
        dtype = self.default_dtype if dtype is None else dtype
        return GPUTensor(self, shape, dtype=dtype, name=name,
                         persist_values=persist_values, allocator=allocator,
                         rounding=self.round_mode)._assign(0)

    def ones(self, shape, dtype=None, name=None, persist_values=True,
             parallel=False, distributed=False, allocator=drv.mem_alloc):
        """
        Returns an array of the given shape and dtype filled with 1's.
        """
        dtype = self.default_dtype if dtype is None else dtype
        return GPUTensor(self, shape, dtype=dtype, name=name,
                         persist_values=persist_values, allocator=allocator,
                         rounding=self.round_mode)._assign(1)

    def empty_like(self, other_ary, name=None):
        """
        Returns an array with the same params as another
        """
        return GPUTensor(self, other_ary.shape, dtype=other_ary.dtype,
                         name=name, persist_values=other_ary.persist_values,
                         allocator=other_ary.allocator, rounding=self.round_mode)

    def zeros_like(self, other_ary, name=None):
        """
        Returns an array with the same params as another
        """
        return GPUTensor(self, other_ary.shape, dtype=other_ary.dtype,
                         name=name, persist_values=other_ary.persist_values,
                         allocator=other_ary.allocator,
                         rounding=self.round_mode)._assign(0)

    def compensated_sum(self, sum_tensor, cmp_tensor, add_tensor, cmp_scale=1.0, add_scale=1.0):
        from neon.backends.float_ew import _get_compensated_sum_kernel, _get_fast_ew_dims

        if cmp_tensor.kahan_reset and cmp_tensor.kahan_count > cmp_tensor.kahan_reset:
            cmp_scale = 0
            cmp_tensor.kahan_count = 0

        assert sum_tensor.dtype.type == cmp_tensor.dtype.type == add_tensor.dtype.type

        cmp_tensor.kahan_count += 1

        shape, strides = _get_fast_ew_dims(sum_tensor.size)

        kernel = _get_compensated_sum_kernel(
            sum_tensor.dtype.str[1:], sum_tensor.rounding > 0)

        kernel.prepared_async_call(
            (shape[0], 1, 1), (32, 1, 1), self.stream, self._get_rand_state_dev(),
            sum_tensor.gpudata, cmp_tensor.gpudata, add_tensor.gpudata,
            cmp_scale, add_scale,
            strides[0], strides[1],
            shape[1], sum_tensor.rounding)

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

    def cublas_dot(self, A, B, C, alpha=1.0, beta=0.0):
        """
        Matrix multiplication using cublas library. Intended for use on Kepler
        GPUs where maxas kernels are not supported.

        C = alpha * (AB) + beta * C

        Arguments:
            A (Tensor): Input tensor
            B (Tensor): Input tensor
            C (Tensor): Output tensor
            alpha (float): Scalar for AB
            beta (float): Scalar for C
        """
        lda = max(A.strides)
        ldb = max(B.strides)
        ldc = max(C.strides)

        opA = 't' if A.is_trans else 'n'
        opB = 't' if B.is_trans else 'n'

        m = A.shape[0]
        n = B.shape[1]
        k = A.shape[1]

        # Swap A and B to map from C order to Fortran
        if A.dtype == np.float32:
            cublas.cublasSgemm(self.cublas_handle, opB, opA, n, m, k, alpha, B.gpudata,
                               ldb, A.gpudata, lda, beta, C.gpudata, ldc)
        elif A.dtype == np.float16:
            #fp16 gemm not supported by cublas until 7.5, so do conversion
            A_temp = self._buf_malloc((A.shape[0], A.shape[1] * 2))
            B_temp = self._buf_malloc((B.shape[0], B.shape[1] * 2))
            C_temp = self._buf_malloc((C.shape[0], C.shape[1] * 2))

            A_fp32 = GPUTensor(self, A.shape, dtype=np.float32, gpudata=A_temp.gpudata,
                               strides=A.strides, is_trans=A.is_trans)
            B_fp32 = GPUTensor(self, B.shape, dtype=np.float32, gpudata=B_temp.gpudata,
                               strides=B.strides, is_trans=B.is_trans)
            C_fp32 = GPUTensor(self, C.shape, dtype=np.float32, gpudata=C_temp.gpudata,
                               strides=C.strides, is_trans=C.is_trans)

            A_fp32[:] = A
            B_fp32[:] = B
            C_fp32[:] = C
            cublas.cublasSgemm(self.cublas_handle, opB, opA, n, m, k, alpha, B_fp32.gpudata,
                               ldb, A_fp32.gpudata, lda, beta, C_fp32.gpudata, ldc)
            C[:] = C_fp32

            self._buf_free()
        else:
            raise TypeError("Unsupported type for cublas gemm")

    def copy_transpose(self, a, out, axes=None, repeat=1):
        """
        Function to perform a fast copy transpose/dimshuffle operation.
        Works just like numpy.transpose, but requires an output tensor argument.
        """
        assert a.dtype == out.dtype
        assert a.size == out.size
        assert a.gpudata != out.gpudata

        if axes is None:
            axes = tuple(range(len(a.shape)-1,-1,-1))
        elif type(axes) is not tuple:
            axes = tuple(axes)

        assert all(out.shape[i]==a.shape[x] for i,x in enumerate(axes))

        from neon.backends.convolution import _get_copy_transpose_kernel

        kernel = _get_copy_transpose_kernel(a.dtype.str, a.shape, axes)

        args = kernel.args + a.strides + out.strides

        # Warmup
        if repeat > 1:
            for r in range(max(repeat // 10, 1)):
                kernel.prepared_async_call(kernel.grid, kernel.block,
                    self.stream, out.gpudata, a.gpudata, *args)

        if self.bench > 1 or repeat > 1:
            start, end = _get_events()
            start.record(self.stream)

        for r in range(repeat):
            kernel.prepared_async_call(kernel.grid, kernel.block,
                self.stream, out.gpudata, a.gpudata, *args)

        if self.bench > 1 or repeat > 1:
            end.record(self.stream)
            end.synchronize()
            msecs = end.time_since(start) / repeat
            bandwidth = a.nbytes*2 / (msecs * 1024 * 1024)
            print("%7.3f msecs %4.0f GBps copy_transpose" % (msecs, bandwidth))

    def init_mark(self):
        """
        Generate a timing mark object

        Returns:
            timing mark (pycude driver event)
        """
        return drv.Event()

    def record_mark(self, marker):
        """
        Mark the current time

        Arguments:
            marker (time mark): timing mark generated by init_mark()
        """
        marker.record(self.stream)

    def synchronize_mark(self, marker):
        """
        Synchronize on the given marker

        Arguments:
            marker (time mark): timing mark generated by init_mark()
        """
        marker.synchronize()

    def get_time(self, start, end):
        """
        Return time between start and end marks

        Arguments:
            start (time maker): start time mark

            end (time marker): end time mark

        Returns:
            time elapsed between start and end time marks in milliseconds
        """
        return end.time_since(start)


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
