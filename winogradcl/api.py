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

kernels = {}

def shuffle(ctx, queue, src, src_shape, dst):
    global kernels

    # will shuffle src into dst, transposing first and last dimensions
    # dimensions are taken to be:
    # A B C
    # where B is product of the dimensions other than first and last
    A = src_shape[0]
    C = src_shape[-1]
    B = functools.reduce(mul, src_shape[1:-1])

    grid = (ceil_div(C, 32), ceil_div(A, 32), B)
    block = (32, 8, 1)
    kernel_name = 'shuffle_f4'
    kernel = kernels.get(kernel_name, None)
    if kernel is None:
        kernel = get_shuffle_kernel_d3_cl(ctx, 'f4')
        kernels[kernel_name] = kernel

    call_cl_kernel(kernel, queue,
        grid, block,
        dst, src,
        B * C, C,
        A * B, A)

# def fprop(ctx, queue, I, I_layout, W, W_layout, O, O_layout):
def fprop(ctx, queue, I, I_shape, W, W_shape, O, O_shape):
    """
    layout should be:
    - for I:  'C H W N'
    - for W:  'Ci H W Co'
    - for O:  'C H W N'
    """

