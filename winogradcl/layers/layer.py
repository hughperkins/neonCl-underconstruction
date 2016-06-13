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
import numpy as np


class Layer(object):

    """
    Top level generic neural network layer class from which all other layer
    types inherit.

    Arguments:
        name (string): Name identifying this layer (in logs, etc.)
        parallelism (int): Type of parallelism preferred by this layer. Possible
            values are "Unknown", "Disabled", and "Data". Only applicable to
            distributed backends (see gen_backend for details).
    """

    def __init__(self, be, name=None):
#        super(Layer, self).__init__(name)
        self.outputs = None
        self.has_params = False
        self.inputs = None
        self.owns_output = True
        self.owns_delta = False
        self.deltas = None
        self.actual_bsz = None
        self.be = be

    def configure(self, in_obj):
        """
        sets shape based parameters of this layer given an input tuple or int
        or input layer

        Arguments:
            in_obj (int, tuple, Layer or Tensor or dataset): object that provides shape
                                                             information for layer

        Returns:
            (tuple): shape of output data
        """
        if isinstance(in_obj, Layer):
            self.prev_layer = in_obj
            self.in_shape = in_obj.out_shape
            if self.parallelism == "Unknown":
                self.parallelism = in_obj.parallelism
        else:
            self.prev_layer = None
            self.in_shape = in_obj  # input is a shape tuple or int directly

    def set_deltas(self, delta_buffers):
        """
        Use pre-allocated (by layer containers) list of buffers for backpropagated error.
        Only set deltas for layers that own their own deltas
        Only allocate space if layer owns its own deltas (i.e. bias, activation work in-place,
        so do not own their deltas).

        Arguments:
            delta_buffers (list): list of pre-allocated tensors (provided by layer container)

        """
        if self.next_layer is not None and self.next_layer.parallelism != self.parallelism:
            self.owns_delta = True

        if self.owns_delta and self.prev_layer:
            if type(self.prev_layer) in (BranchNode, ColorNoise):
                self.deltas = self.prev_layer.deltas
            else:
                self.deltas = self.be.iobuf(self.in_shape, shared=delta_buffers[0],
                                            parallelism=self.parallelism)
                delta_buffers.reverse()
        else:
            self.deltas = None

    def fprop(self, inputs, inference=False):
        """
        Apply the forward pass transformation to the input data.

        Arguments:
            inputs (Tensor): input data

        Returns:
            Tensor: output data
        """
        raise NotImplementedError

    def _fprop_inference(self, inputs):
        """
        Apply the forward pass transformation to the input data.

        May skip any computation not needed for doing inference only.

        Calling bprop subsequently is not valid.

        Arguments:
            inputs (Tensor): input data

        Returns:
            Tensor: output data
        """
        raise NotImplementedError

    def bprop(self, error):
        """
        Apply the backward pass transformation to the input data.

        Arguments:
            error (Tensor): deltas back propagated from the adjacent higher layer

        Returns:
            Tensor: deltas to propagate to the adjacent lower layer
        """
        raise NotImplementedError


class ParameterLayer(Layer):

    """
    Intermediate class used for common functionality for any layer with weights.

    Not intended to be used directly.

    Arguments:
        init (Initializer, optional): Initializer object to use for
            initializing layer weights
        name (str, optional): layer name. Defaults to "ParameterLayer"
    """

    def __init__(self, be, init=None, name=None):
        super(ParameterLayer, self).__init__(be=be, name=name)
        self.has_params = True
        self.init = init
        self.W = None
        self.dW = None
        self.weight_shape = None
        self.batch_sum = None
        self.batch_sum_shape = None
        self.states = []
        self.owns_delta = True

    @classmethod
    def gen_class(cls, pdict):
        if 'init' in pdict and pdict['init'] is not None:
            cname = pdict['init']['type']
            icls = load_class(cname)
            init = icls(**pdict['init']['config'])
            pdict['init'] = init
        return cls(**pdict)


class Convolution(ParameterLayer):

    """
    Convolutional layer implementation.

    Arguments:
        fshape (tuple(int)): three dimensional shape of convolution window
        strides (int, dict, optional): strides to apply convolution
            window over. An int applies to both dimensions, or a dict with
            str_h and str_w applies to h and w dimensions distinctly.  Defaults
            to str_w = str_h = None
        padding (int, dict, optional): padding to apply to edges of
            input. An int applies to both dimensions, or a dict with pad_h
            and pad_w applies to h and w dimensions distinctly.  Defaults
            to pad_w = pad_h = None
        init (Initializer, optional): Initializer object to use for
            initializing layer weights
        name (str, optional): layer name. Defaults to "ConvolutionLayer"
    """

    def __init__(self, fshape, be, strides={}, padding={}, init=None, bsum=False,
                 name=None):
        super(Convolution, self).__init__(init=init, name=name, be=be)
        self.nglayer = None
        bsum = bsum and not self.be.deterministic
        self.convparams = {'str_h': 1, 'str_w': 1, 'str_d': 1,
                           'pad_h': 0, 'pad_w': 0, 'pad_d': 0,
                           'T': 1, 'D': 1, 'bsum': bsum}  # 3D paramaters

        # keep around args in __dict__ for get_description.
        self.fshape = fshape
        self.strides = strides
        self.padding = padding
        self.bsum = bsum

        if isinstance(fshape, tuple) or isinstance(fshape, list):
            fkeys = ('R', 'S', 'K') if len(fshape) == 3 else ('T', 'R', 'S', 'K')
            fshape = {k: x for k, x in zip(fkeys, fshape)}
        if isinstance(strides, int):
            strides = {'str_h': strides, 'str_w': strides}
        if isinstance(padding, int):
            padding = {'pad_h': padding, 'pad_w': padding}
        for d in [fshape, strides, padding]:
            self.convparams.update(d)

    def configure(self, in_obj):
        super(Convolution, self).configure(in_obj)
        if self.nglayer is None:
            assert isinstance(self.in_shape, tuple)
            ikeys = ('C', 'H', 'W') if len(self.in_shape) == 3 else ('C', 'D', 'H', 'W')
            shapedict = {k: x for k, x in zip(ikeys, self.in_shape)}
            shapedict['N'] = self.be.bsz
            self.convparams.update(shapedict)
            self.nglayer = self.be.conv_layer(self.be.default_dtype, **self.convparams)
            (K, M, P, Q, N) = self.nglayer.dimO
            self.out_shape = (K, P, Q) if M == 1 else (K, M, P, Q)
        if self.weight_shape is None:
            self.weight_shape = self.nglayer.dimF2  # (C * R * S, K)
        if self.convparams['bsum']:
            self.batch_sum_shape = (self.nglayer.K, 1)
        return self

    def fprop(self, inputs, inference=False, beta=0.0):
        self.inputs = inputs
        self.be.fprop_conv(self.nglayer, inputs, self.W, self.outputs, beta=beta,
                           bsum=self.batch_sum)
        return self.outputs

    def bprop(self, error, alpha=1.0, beta=0.0):
        if self.deltas:
            self.be.bprop_conv(self.nglayer, error, self.W, self.deltas,
                               alpha=alpha, beta=beta)
        self.be.update_conv(self.nglayer, self.inputs, error, self.dW)
        return self.deltas



