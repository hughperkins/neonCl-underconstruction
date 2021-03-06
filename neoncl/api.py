"""
api for running OpenCL ports of nervana neon convolutional kernels

status: in progress

approximate guidelines/requirements:
- caller should handle opencl context and queue setup
- caller should allocate cl buffers
- library can/should provide a means to provide required dimensions of buffers to caller
- library will check dimensions of incoming buffers
"""
from neoncl.backends.kernels.cl.clshuffler import get_shuffle_kernel_d3_cl
from neoncl.backends.kernels.cl.callkernel import call_cl_kernel
from neoncl.util.math_helper import ceil_div
import numpy as np
import pyopencl as cl
from operator import mul
import functools
from neoncl.backends.convolution import FpropCuda, BpropCuda, UpdateCuda


mf = cl.mem_flags

def output_dim(caffe_compat, X, S, padding, stride):
    """
    compute along 1 dimension, with these sizes, what will be the output dimension

    Arguments:
        X (int): input data dimension
        S (int): filter dimension
        padding (int): padding on each side
        stride (int): striding
    """

    if caffe_compat:
        size = int(ceil(float(X - S + 2 * padding) // stride)) + 1
        if padding > 0 and (size - 1)*stride >= X + padding:
            # decrement size if last pooling op is completely in padding
            size -= 1
    else:
        # normal neon output size determination
        size = (X - S + 2 * padding) // stride + 1

    return size


class Shuffler(object):
    # will shuffle src into dst, transposing first and last dimensions
    # dimensions are taken to be:
    # A B C
    # where B is product of the dimensions other than first and last
    def __init__(self, ctx, src_shape):
        self.kernel = get_shuffle_kernel_d3_cl(ctx, 'f4')
        self.A = src_shape[0]
        self.C = src_shape[-1]
        self.B = functools.reduce(mul, src_shape[1:-1])
        self.grid = (ceil_div(self.C, 32), ceil_div(self.A, 32), self.B)
        self.block = (32, 8, 1)
        self.BC = self.B * self.C
        self.AB = self.A * self.B

    def shuffle(self, queue, dst, src):
        call_cl_kernel(
            self.kernel, queue,
            self.grid, self.block,
            dst, src,
            self.BC, self.C,
            self.AB, self.A)


class Convolver(object):
    def __init__(self, ctx, N, Ci, Co, kH, kW, iH, iW, padH, padW, dH, dW):
        """
        layout should be:
        - for I:  'C H W N'
        - for W:  'Ci H W Co'
        - for O:  'C H W N'
        """
        self.ctx = ctx
        self.Ci = Ci
        self.Co = Co
        self.iH = iH
        self.iW = iW
        self.N = N
        self.kH= kH
        self.kW = kW
        self.dH = dH
        self.dW = dW
        oH = output_dim(False, iH, kH, padH, dH)
        oW = output_dim(False, iW, kW, padW, dW)

        assert padH == padW
        self.fpropcuda = FpropCuda(ctx, 'f4',
            N, Ci, Co,
            1, iH, iW,
            1, kH, kW,
            1, oH, oW,
            0, padH, padW,
            0, dH, dW)

        self.bpropcuda = BpropCuda(ctx, 'f4',
            N, Ci, Co,
            1, iH, iW,
            1, kH, kW,
            1, oH, oW,
            0, padH, padW,
            0, dH, dW)

        self.updatecuda = UpdateCuda(ctx, 'f4',
            N, Ci, Co,
            1, iH, iW,
            1, kH, kW,
            1, oH, oW,
            0, padH, padW,
            0, dH, dW)

    def getILayout(self):
        return 'Ci iH iW N'

    def getGradILayout(self):
        return 'Ci iH iW N'

    def getWLayout(self):
        return 'Ci kH kW Co'

    def getGradWLayout(self):
        return 'Ci kH kW Co'

    def getOLayout(self):
        return 'Co oH oW N'

    def getGradOLayout(self):
        return 'Co oH oW N'

    def getScratchSize(self, fpropOnly=False):
        if fpropOnly:
            return 0
        return self.getBpropGradIScratchSize()

    def getFpropScratchSize(self):
        return 0

    def getBpropGradWScratchSize(self):
        return 0

    def getBpropGradIScratchSize(self):
        return self.Ci * self.Co * self.kH * self.kW

    def getIShape(self):
        return (self.Ci, self.iH, self.iW, self.N)

    def getGradIShape(self):
        return self.getIShape()

    def getWShape(self):
        return (self.Ci, self.kH, self.kW, self.Co)

    def getGradWShape(self):
        return self.getWShape()

    def getOShape(self):
        return (self.Co, self.oH, self.oW, self.N)

    def getGradOShape(self):
        return self.getOShape()

    def fprop(self, queue, I, W, O, scratch=None):
        self.fpropcuda.bind_params(I, W, O, 1.0, 0.0)
        self.fpropcuda.execute(queue)

    def bprop_gradW(self, queue, I, gradO, gradW, scratch=None):
        self.updatecuda.bind_params(I, gradO, gradW, 1.0)
        self.updatecuda.execute(queue)

    def bprop_gradI(self, queue, gradO, W, gradI, scratch):
        Wt = scratch
        self.bpropcuda.shuffle(queue, Wt, W)
        self.bpropcuda.bind_params(gradO, Wt, gradI, 1.0, 0.0)
        self.bpropcuda.execute(queue)

