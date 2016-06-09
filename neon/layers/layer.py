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
import logging
import numpy as np

from neon import NervanaObject
#from neon.backends import Autodiff
from neon.backends.backend import Tensor
from neon.util.persist import load_class


logger = logging.getLogger(__name__)


def interpret_in_shape(xshape):
    """
    Helper function to interpret the tensor layout of preceding layer to handle non-recurrent,
    recurrent, and local layers
    """
    if isinstance(xshape, int):
        return (xshape, 1)
    else:
        if len(xshape) == 2:
            return xshape
        else:
            return (np.prod(xshape), 1)


class Layer(NervanaObject):

    """
    Top level generic neural network layer class from which all other layer
    types inherit.

    Arguments:
        name (string): Name identifying this layer (in logs, etc.)
        parallelism (int): Type of parallelism preferred by this layer. Possible
            values are "Unknown", "Disabled", and "Data". Only applicable to
            distributed backends (see gen_backend for details).
    """

    def __init__(self, name=None, parallelism="Unknown"):
        super(Layer, self).__init__(name)
        self.outputs = None
        self.has_params = False
        self.inputs = None
        self.owns_output = True
        self.owns_delta = False
        self.deltas = None
        self.parallelism = parallelism
        self.revert_list = []
        self.next_layer = None
        self.actual_bsz = None
        self.actual_seq_len = None

    def __str__(self):
        """
        Format the layer as a printable string.
        """
        ret = '{} {}'.format(self.classnm, self.name)
        return ret

    def nested_str(self, level=0):
        """
        Utility function for displaying layer info with a given indentation level

        Arguments:
            level (int, optional): indentation level
        """

        return "  " * level + str(self)

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
            if isinstance(in_obj, (tuple, int, list)):
                self.in_shape = in_obj  # input is a shape tuple or int directly
            elif isinstance(in_obj, Tensor):
                self.in_shape = (in_obj.shape[0], in_obj.shape[1] / self.be.bsz)
            else:
                self.in_shape = in_obj.shape  # This is a dataset

    def allocate(self, shared_outputs=None):
        """
        Allocates output buffer to store activations from fprop.
        Don't reallocate if it already exists.
        Only allocate space if layer owns its own output (i.e. bias, activation work in-place,
        so do not own their output).
        outputs can be allocated from a pre-allocated pool if shared_outputs is provided

        Arguments:
            shared_outputs (Tensor, optional): pre-allocated tensor for activations to be
                                               computed into

        """
        if self.outputs:
            return
        if self.owns_output:
            self.outputs = self.be.iobuf(self.out_shape, shared=shared_outputs,
                                         parallelism=self.parallelism)

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

    def set_next(self, layer):
        self.next_layer = layer

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

    def get_terminal(self):
        """
        Used for recursively getting final nodes from layer containers
        """
        return self

    def serialize(self):
        """
        Get state parameters for this layer

        Returns:
            ?: whatever data this model wants to receive in order to restore state
        """
        if self.has_params:
            return self.get_params()

    def load_weights(self, pdict, load_states=True):
        self.set_params(pdict)
        if load_states:
            self.set_states(pdict)

    def get_param_attrs(self):
        return dict(parallel=(self.parallelism in ("Data", "Model")),
                    distributed=(self.parallelism == "Model"))

    def set_params(self, pdict):
        pass

    def set_states(self, pdict):
        pass

    def set_batch_size(self, N):
        self.actual_bsz = N

    def set_seq_len(self, S):
        self.actual_seq_len = S

    def get_description(self, **kwargs):
        """
        Get layer parameters. All parameters are needed for optimization, but
        only Weights are serialized.

        Arguments:
        """
        return super(Layer, self).get_description(**kwargs)


class ParameterLayer(Layer):

    """
    Intermediate class used for common functionality for any layer with weights.

    Not intended to be used directly.

    Arguments:
        init (Initializer, optional): Initializer object to use for
            initializing layer weights
        name (str, optional): layer name. Defaults to "ParameterLayer"
    """

    def __init__(self, init=None, name=None,
                 parallelism="Unknown"):
        super(ParameterLayer, self).__init__(name, parallelism)
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

    def allocate(self, shared_outputs=None):
        super(ParameterLayer, self).allocate(shared_outputs)
        if self.W is None:
            self.init_params(self.weight_shape)
        if self.batch_sum_shape is not None:
            self.batch_sum = self.be.empty(self.batch_sum_shape, dtype=np.float32,
                                           **self.get_param_attrs())

    def init_params(self, shape):
        """
        Allocate layer parameter buffers and initialize them with the
            supplied initializer.

        Arguments:
            shape (int, tuple): shape to allocate for layer parameter
                buffers.
        """
        self.W = self.be.empty(shape, **self.get_param_attrs())
        self.dW = self.be.empty_like(self.W)
        self.states = []

        if isinstance(self.init, Tensor) or isinstance(self.init, np.ndarray):
            assert self.init.shape == self.W.shape, "Initial weights shape does not match"
            self.W[:] = self.init
        else:
            self.init.fill(self.W)

    def get_params(self):
        """
        Get layer parameters, gradients, and states for optimization
        """
        return ((self.W, self.dW), self.states)

    def get_params_serialize(self, keep_states=True):
        return self.get_description(get_weights=True, keep_states=keep_states)

    def get_description(self, get_weights=False, keep_states=True):
        """
        Get layer parameters. All parameters are needed for optimization, but
        only Weights are serialized.

        Arguments:
            keep_states (bool): Control whether all parameters are returned
                or just weights for serialization. Defaults to True.
        """
        serial_dict = super(ParameterLayer, self).get_description()
        if get_weights:
            serial_dict['params'] = {'W': self.W.get()}
            if keep_states:
                serial_dict['states'] = [s.get() for s in self.states]
        return serial_dict

    def set_params(self, pdict):
        """
        Set layer parameters (weights). Allocate space for other parameters but do not initialize
        them.

        Arguments:
            pdict (dict, ndarray): dictionary or ndarray with layer parameters
                                   [support for ndarray is DEPRECATED and will be removed]
        """
        assert type(pdict) is dict
        for key in pdict['params']:
            if not hasattr(self, key):
                setattr(self, key, None)

            attr = getattr(self, key)
            if isinstance(attr, Tensor):
                # this attr has already been allocated
                # get set the values
                attr.set(pdict['params'][key])
            elif type(pdict['params'][key]) is np.ndarray:
                setattr(self, key, self.be.array(pdict['params'][key], **self.get_param_attrs()))
            else:
                setattr(self, key, pdict['params'][key])

        if self.dW is None:
            self.dW = self.be.empty_like(self.W)

    def set_states(self, pdict):
        if 'states' not in pdict:
            # if states was not serialized then leave
            # this empty, the optimizer will initialize it
            self.states = []
        else:
            # this needs to be done in two steps for MGPU backend
            if self.states is None or len(self.states) == 0:
                self.states = [self.be.zeros_like(self.dW)
                               for i in range(len(pdict['states']))]

            for ind in range(len(pdict['states'])):
                self.states[ind].set(pdict['states'][ind])


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

    def __init__(self, fshape, strides={}, padding={}, init=None, bsum=False,
                 name=None, parallelism="Data"):
        super(Convolution, self).__init__(init, name, parallelism)
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

    def __str__(self):
        spatial_dim = len(self.in_shape[1:])
        spatial_str = "%d x (" + "x".join(("%d",) * spatial_dim) + ")"
        padstr_str = ",".join(("%d",) * spatial_dim)
        padstr_dim = ([] if spatial_dim == 2 else ['d']) + ['h', 'w']

        pad_tuple = tuple(self.convparams[k] for k in ['pad_' + d for d in padstr_dim])
        str_tuple = tuple(self.convparams[k] for k in ['str_' + d for d in padstr_dim])

        fmt_tuple = (self.name,) + self.in_shape + self.out_shape + pad_tuple + str_tuple
        fmt_string = "Convolution Layer '%s': " + \
                     spatial_str + " inputs, " + spatial_str + " outputs, " + \
                     padstr_str + " padding, " + padstr_str + " stride"

        return ((fmt_string % fmt_tuple))

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
            self.be.bprop_conv(self.nglayer, self.W, error, self.deltas,
                               alpha=alpha, beta=beta)
        self.be.update_conv(self.nglayer, self.inputs, error, self.dW)
        return self.deltas


class CompoundLayer(list):
    """
    Base class for macro layers.
    """
    def __init__(self, bias=None, batch_norm=False, activation=None, name=None):
        super(CompoundLayer, self).__init__()
        if batch_norm and (bias is not None):
            raise AttributeError('Batchnorm and bias cannot be combined')
        self.activation = activation
        self.batch_norm = batch_norm
        self.bias = bias
        self.base_name = name

    @classmethod
    def gen_class(cls, pdict):
        for key in ['init', 'bias', 'activation']:
            if key in pdict and pdict[key] is not None:
                cname = pdict[key]['type']
                icls = load_class(cname)
                if 'config' not in pdict[key]:
                    pdict[key]['config'] = {}
                pdict[key] = icls(**pdict[key]['config'])
        return cls(**pdict)

    def init_base_name(self):
        if self.base_name is None:
            self.base_name = self[-1].name

    def add_postfilter_layers(self):
        self.init_base_name()
        if self.bias is not None:
            name = self.base_name+'_bias'
            self.append(Bias(init=self.bias, name=name))
        if self.batch_norm:
            name = self.base_name+'_bnorm'
            self.append(BatchNorm(name=name))
        if self.activation is not None:
            name = self.base_name + '_' + self.activation.classnm
            self.append(Activation(transform=self.activation, name=name))


class Conv(CompoundLayer):

    """
    A convolutional layer with a learned bias and activation, implemented as a
    list composing separate Convolution, Bias and Activation layers.

    Arguments:
        fshape (tuple(int)): three dimensional shape of convolution window
        init (Initializer, optional): Initializer object to use for
            initializing layer weights and bias
        strides (int, dict, optional): strides to apply convolution
            window over. An int applies to both dimensions, or a dict with
            str_h and str_w applies to h and w dimensions distinctly.  Defaults
            to str_w = str_h = None
        pad (int, dict, optional): padding to apply to edges of
            input. An int applies to both dimensions, or a dict with pad_h
            and pad_w applies to h and w dimensions distinctly.  Defaults
            to pad_w = pad_h = None
        bias (Initializer): an initializer to use for bias parameters
        activation (Transform): a transform object with fprop and bprop
            functions to apply
        name (str): the root name for the layer, suffixes are automatically
            generated for the component layers

    """

    def __init__(self, fshape, init, strides={}, padding={},
                 bias=None,
                 batch_norm=False,
                 activation=None,
                 name=None):
        super(Conv, self).__init__(bias=bias, batch_norm=batch_norm,
                                   activation=activation, name=name)
        self.append(Convolution(fshape=fshape, strides=strides, padding=padding,
                                init=init, bsum=batch_norm,
                                name=name))
        self.add_postfilter_layers()



