"""
api for running convolutions using winograd

status: in progress

approximate guidelines/requirements:
- caller should handle opencl context and queue setup
- caller should allocate cl buffers
- library can/should provide a means to provide required dimensions of buffers to caller
- library will check dimensions of incoming buffers
"""
# from winogradcl.backends.kernels.cl import ShuffleRunner
from winogradcl.backends.kernels.cl.clshuffler import get_shuffle_kernel_d3_cl
from winogradcl.backends.kernels.cl.callkernel import call_cl_kernel
from winogradcl.util.math_helper import ceil_div
import numpy as np
from operator import mul
import functools


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


# def fprop(ctx, queue, I, I_layout, W, W_layout, O, O_layout):
def fprop(ctx, queue, I, I_shape, W, W_shape, O, O_shape):
    """
    layout should be:
    - for I:  'C H W N'
    - for W:  'Ci H W Co'
    - for O:  'C H W N'
    """
    Ci = W_shape[0]
    Co = W_shape[3]
    kH = W_shape[1]
    kW = W_shape[2]
    iH = I_shape[1]
    iW = I_shape[2]
    padH = kH // 2
    padW = kW // 2
    assert padH == padW
    conv = Convolution((kH, kW, Co), strides=1, padding=1, be=be) #, init=init)
    conv.configure((Ci, iH, iW))
    conv.W = W
    conv.outputs = O

def bprop_gradW(ctx, queue, gradO, gradO_shape, W, W_shape, gradW, gradW_shape):
    pass

def bprop_gradI(ctx, queue, gradO, gradO_shape, W, W_shape, gradI, gradI_shape):
    pass

