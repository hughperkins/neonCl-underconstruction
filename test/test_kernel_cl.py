import pyopencl as cl
from neoncl.backends.kernels.cl.convolution_cl import _get_conv_kernel
import winograd_kernels_cl
import types


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

kernel = winograd_kernels_cl.fprop_filter_trans_4x4_kernel(ctx=ctx)
kernel = winograd_kernels_cl.xprop_image_trans_4x4_kernel(ctx=ctx)

for kernel_name in dir(winograd_kernels_cl):
    attr = winograd_kernels_cl.__dict__[kernel_name]
    if isinstance(attr, types.FunctionType):
        print(kernel_name, type(attr))
        attr(ctx=ctx)

