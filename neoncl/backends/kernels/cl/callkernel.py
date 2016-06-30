import pyopencl as cl
import numpy as np


def call_cl_kernel(kernel, queue, grid, block, *args):
    print('grid', grid, 'block', block, 'kernel', kernel)
    blockDim = len(block)
    if blockDim == 3:
        globalSize = (block[0] * grid[0], block[1] * grid[1], block[2] * grid[2])  # hacky? what do you mean? :-P
    else:
        raise Exception('not implemented')

    newargs = []
    i = 0
    for arg in args:
        if isinstance(arg, int):
            newargs.append(np.int32(arg))
        elif isinstance(arg, float):
            newargs.append(np.float32(arg))
        elif isinstance(arg, cl.cffi_cl.Buffer):
            newargs.append(arg)
        elif isinstance(arg, cl.cffi_cl.LocalMemory):
            newargs.append(arg)
        else:
            raise Exception('type not implemented %s' % type(arg))
        i += 1
    kernel(queue, globalSize, block, *newargs)

