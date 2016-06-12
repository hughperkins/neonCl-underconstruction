import pyopencl as cl
import numpy as np

mf = cl.mem_flags

class MyClTensor(object):
    def __init__(self, be, gpudata, shape, size):
        self.be = be
        self.gpudata = gpudata
        self.size = size
        self.dtype = np.float32
        self.cpudata = None
        self.shape = shape

    @staticmethod
    def from_np(be, np_data):
        clbuf = cl.Buffer(be.cl_ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np_data)

        tensor = MyClTensor(be, clbuf, shape=np_data.shape, size=np_data.size)
        tensor.cpudata = np_data
        return tensor

    def to_host(self):
        if self.cpudata is None:
            raise Exception('not implemented')
        cl.enqueue_copy(self.be.q, self.cpudata, self.gpudata)


