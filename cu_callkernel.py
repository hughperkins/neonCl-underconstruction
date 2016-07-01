import numpy as np
from pycuda import gpuarray


def call_cu_kernel(kernel, grid, block, *args):
    pass
    newargs = []
    i = 0
    for arg in args:
        if isinstance(arg, int):
            newargs.append(np.int32(arg))
        elif isinstance(arg, float):
            newargs.append(np.float32(arg))
        elif isinstance(arg, gpuarray.GPUArray):
            newargs.append(arg)
        #elif isinstance(arg, cl.cffi_cl.LocalMemory):
        #    newargs.append(arg)
        else:
            raise Exception('type not implemented %s' % type(arg))
        i += 1

    kernel(grid=grid, block=block, *newargs)

