import pyopencl as cl
from neoncl.backends.kernels.cl.convolution_cl import _get_conv_kernel
from winograd_kernels_cl import get_fprop_filter_trans_kernel

gpu_idx = 0

platforms = cl.get_platforms()
i = 0
for platform in platforms:
   gpu_devices = platform.get_devices(device_type=cl.device_type.GPU)
   if gpu_idx < i + len(gpu_devices):
       ctx = cl.Context(devices=[gpu_devices[gpu_idx-i]])
       break
   i += len(gpu_devices)

#kernel = _get_conv_kernel(ctx=ctx, options='', dtype='f4', filter_size=9, bsum=False, operation='fprop')
#kernel = _get_conv_kernel(ctx=ctx, options='', dtype='f4', filter_size=9, bsum=False, operation='bprop')
#kernel = _get_conv_kernel(ctx=ctx, options='', dtype='f4', filter_size=9, bsum=False, operation='update')
kernel = get_fprop_filter_trans_kernel(ctx=ctx)

