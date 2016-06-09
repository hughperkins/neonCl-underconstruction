# ----------------------------------------------------------------------------
# Copyright 2015 Nervana Systems Inc.
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
Defines Tensor and Backend class
"""

import numpy as np
import logging
from math import ceil

logger = logging.getLogger(__name__)


class Tensor(object):
    """
    The n-dimensional array data structure. GPUTensor and Tensor inherits
    Tensor. Depending on backend, may have additional keyword arguments.
    All non-keywords arguments shall be in exact same order as Tensor.

    Arguments:
        backend (Backend): backend of the tensor.
        shape (tuple, optional): shape of the tensor.
        dtype (numpy.ndtype, optional): underlying data type of the elements.
        name (str, optional): name indentifying the tensor (used in printing).
        persist_values (bool, optional): If set to True (the default), the
                                         values assigned to this Tensor will
                                         persist across multiple begin and
                                         end calls.  Setting to False may
                                         provide a performance increase if
                                         values do not need to be maintained
                                         across such calls

    See also:
        GPUTensor class, Tensor class

    Notes:
        Unlike numpy, in this implementation we never collapse dimensions, and
        the minimal number of dimensions will be _min_dims (currently set to
        2).  So a wrapped scalar will have dimension 1x1.
    """
    def __init__(self,
                 backend,
                 shape=None,
                 dtype=np.float32,
                 name=None,
                 persist_values=True):

        self.backend = backend
        self.shape = shape
        self.dtype = dtype
        self.name = name
        self.persist_values = persist_values
        self._min_dims = 2
        self.base = None


class Backend(object):
    """
    Backend interface used to manipulate Tensor data. This abstract base class
    defines what operations each concrete backend must support.
    NervanaGPU and NervanaCPU inherit Backend.

    Arguments:
        rng_seed (int, optional): random number generator seed value
        default_dtype (numpy.ndtype, optional): Elemental data type to use when
                                                creating new tensors if not
                                                otherwise specified.  Defaults
                                                to np.float32
        compat_mode (str, optional): Flag to match implementation of other
                                     libraries.  Currently only 'caffe' is
                                     supported, defaults to None.
        deterministic(bool, optional): Flag to use deterministic kernels
                                       where applicable.  This
                                       may cause a small increase in memory
                                       usage and slow down.  Only relevant for GPU
                                       backends.
    """
    def __init__(self, rng_seed=None, default_dtype=np.float32,
                 compat_mode=None, deterministic=None):
        # dtype
        self.default_dtype = default_dtype

        # use RandomState instead of seed
#        self.rng_seed = rng_seed
#        self.rng = self.gen_rng(rng_seed)

        # batch size
        self.bsz = None
        self._min_dims = 2

        if compat_mode is not None:
            if compat_mode == 'caffe':
                self.set_caffe_compat()
            else:
                raise ValueError('%s mode not supported currently' % compat_mode)
        else:
            self.compat_mode = None

        if deterministic is not None:
            logger.warning('deterministic arg is deprecated in favor of specifying random seed')

#        self.deterministic = self.rng_seed is not None
        self.deterministic = False   # ???

    def output_dim(self, X, S, padding, strides, pooling=False):
        """
        compute along 1 dimension, with these sizes, what will be the output dimension

        Arguments:
            X (int): input data dimension
            S (int): filter dimension
            padding (int): padding on each side
            strides (int): striding
            pooling (bool): flag for setting pooling layer size
        """

        if self.check_caffe_compat() and pooling:
            size = int(ceil(float(X - S + 2 * padding)/strides)) + 1
            if padding > 0 and (size - 1)*strides >= X + padding:
                # decrement size if last pooling op is completely in padding
                size -= 1
        else:
            # normal neon output size determination
            size = (X - S + 2 * padding)/strides + 1

        if pooling and padding >= S:
            raise ValueError("Padding dim %d incompatible with filter size %d" % (padding, S))

        return size

    def set_caffe_compat(self):
        """
        Set flag to make layers compatible with caffe in terms of conv and pool
        layer output size determination and dropout layer implementation
        """
        self.compat_mode = 'caffe'

    def check_caffe_compat(self):
        return self.compat_mode == 'caffe'

    def iobuf(self, dim0, x=None, dtype=None, name=None, persist_values=True,
              shared=None, parallelism=None):
        """
        Allocate input and output buffer for layer based on batch size. This
        is used because the layer does not know about the batch size.

        Arguments:
            dim0 (tuple or int): I/O buffer dimension for layer (without the
                                 axis specifying the batch size).
            x (data-type, optional): If present and not None, `x` will be
                                     returned directly. `x` will be not None if
                                     the buffer has already been allocated.
            dtype (data-type, optional): If present, specifies the underlying
                                         type to employ for each element.
            name (str, optional): name indentifying the tensor (used in printing).
            persist_values (bool, optional): If set to True (the default), the
                                             values assigned to this Tensor will
                                             persist across multiple begin and
                                             end calls.  Setting to False may
                                             provide a performance increase if
                                             values do not need to be maintained
                                             across such calls
            shared (buffer, optional): If present will attempt to reuse the memory
                                       in shared to allocate the I/O buffer
            parallelism (str, optional): Indicates type of parallelism (Data,
                                         Model) employed by this buffer.
                                         Ignored on CPU and GPU backends,
                                         defaults to no parallelism.
        Returns:
            Tensor: array object
        """
        if x is not None:
            return x
        if isinstance(dim0, tuple):
            if (len(dim0) == 2):
                bufshape = (dim0[0], dim0[1] * self.bsz)
            else:
                bufshape = (np.prod(dim0), self.bsz)
        else:
            bufshape = (dim0, self.bsz)

        if shared is not None:
            out_tsr = shared if shared.shape == bufshape else shared.share(bufshape)
        else:
            out_tsr = self.empty(bufshape, dtype=dtype, name=name, persist_values=persist_values)

        if persist_values and shared is None:
            out_tsr[:] = 0

        return out_tsr

    def shared_iobuf_size(self, shape, parallelism):
        """
        Computes the backend specific size needed for an iobuf with a specified
        shape that is meant to be shared between layers.

        Arguments:
            shape (tuple): Requested iobuf shape
            parallelism (string): Parallelism of layer requesting this iobuf

        Returns:
            int: Size of required iobuf
        """
        num_dev = 1 if parallelism in ('Data', 'Model') else getattr(self, 'num_dev', 1)
        return num_dev * np.prod(shape)

    def distribute_data(self, tensor, layer_parallelism):
        """
        For backends which support distributed training, this will distribute
        or gather the error or activation tensor depending on the type of
        parallelism used to distribute the layer computation. Currently
        this is only supported by multi-GPU in Nervana cloud.

        Arguments:
            tensor: Tensor containing either activations or errors
            layer_parallelism: Type of parallelism expected by the layer

        Returns:
            Tensor which has been altered by this call or None
        """
        return None

    def revert_tensor(self, tensor):
        """
        Reverts a tensor to its original state after being distributed by
        distribute_data

        Arguments:
            tensor: Tensor to be reverted
        """
        pass

    def execute(self, node):
        """
        Execute the optree. There must be one and only one 'assign' op at the
        top of the optree when execute is called.

        Arguments:
            node (OpTreeNode): The op-tree to execute.
        """
        pass

    def take(self, a, indices, axis, out=None):
        """
        Extract elements based on the indices along a given axis.

        Arguments:
            a (Tensor): the Tensor on which to perform the operation
            indices (Tensor, numpy ndarray): indicies of elements to select
            axis (int, optional): the dimension along which to compute.
                                  If set to None, we will extract over all
                                  dimensions (flattened first)
            out (Tensor, optional): where the result will be stored. If out is
                                    None, only the op-tree will be returned.
        """
        return a.take(indices, axis, out)

    def conv_layer(self, dtype,
                   N, C, K,
                   D=1, H=1, W=1,
                   T=1, R=1, S=1,
                   pad_d=0, pad_h=0, pad_w=0,
                   str_d=1, str_h=1, str_w=1,
                   relu=False, bsum=False):
        """
        Create a new ConvLayer parameter object.
        This is then passed as an argument to all the convolution operations.

        Arguments:
            dtype (data-type, optional): If present, specifies the underlying
                                         type to employ for each element.

            N (int): Number of images in mini-batch
            C (int): Number of input feature maps
            K (int): Number of output feature maps

            D (int, optional): Depth of input image.  Defaults to 1
            H (int, optional): Height of input image.  Defaults to 1
            W (int, optional): Width of input image.  Defaults to 1

            T (int, optional): Depth of filter kernel.  Defaults to 1
            R (int, optional): Height of filter kernel.  Defaults to 1
            S (int, optional): Width of filter kernel.  Defaults to 1

            pad_d (int, optional): amount of zero-padding around the depth edge
                                   Defaults to 0.
            pad_h (int, optional): amount of zero-padding around the height edge
                                   Defaults to 0.
            pad_w (int, optional): amount of zero-padding around the width edge
                                   Defaults to 0.

            str_d (int, optional): factor to step the filters by in the depth
                                   direction.  Defaults to 1
            str_h (int, optional): factor to step the filters by in the depth
                                   direction.  Defaults to 1
            str_w (int, optional): factor to step the filters by in the depth
                                   direction.  Defaults to 1

            relu (bool, optional): apply a relu transform to the output for
                                   fprop or bprop.  Defaults to False

            bsum (bool, optional): calculate the sum along the batchnorm axis
                                   for fprop or bprop.  Outputs an fp32 tensor
                                   of size Kx1.  Defaults to False.
        """
        raise NotImplementedError()

    def fprop_conv(self, layer, I, F, O, alpha=1.0, relu=False, repeat=1):
        """
        Forward propagate the inputs of a convolutional network layer to
        produce output

        Arguments:
            layer: the conv layer as a parameter object
            I (Tensor): inputs
            F (Tensor): the weights (filters)
            O (Tensor): outputs
            alpha (float, optional): linear scaling.  Defaults to 1.0
            relu (bool, optional): apply ReLu before output.  Default not to.
            repeat (int, optional): Repeat this operation the specified number
                                    of times.  Defaults to 1.
        """
        raise NotImplementedError()

    def bprop_conv(self, layer, F, E, grad_I, alpha=1.0, repeat=1):
        """
        Backward propagate the error through a convolutional network layer.

        Arguments:
            layer: the conv layer as a parameter object
            F (Tensor): the weights (filters)
            E (Tensor): errors
            grad_I (Tensor): gradient to inputs (output delta)
            alpha (float, optional): linear scaling.  Defaults to 1.0
            repeat (int, optional): Repeat this operation the specified number
                                    of times.  Defaults to 1.
        """
        raise NotImplementedError()

    def update_conv(self, layer, I, E, grad_F, alpha=1.0, repeat=1):
        """
        Compute the updated gradient for a convolutional network layer.

        Arguments:
            layer: the conv layer as a parameter object
            I (Tensor): the inputs
            E (Tensor): the errors
            grad_F (Tensor): filter gradients (weights) to update.
            alpha (float, optional): linear scaling.  Defaults to 1.0
            repeat (int, optional): Repeat this operation the specified number
                                    of times.  Defaults to 1.
        """
        raise NotImplementedError()


