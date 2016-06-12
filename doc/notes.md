# neon

These are my own notes to understand the naming conventions used in neon convolution.

## base

In base neon, not modified one:

### fprop

### bprop gradI

```
layers.layer.bprop():
    be.brop_conv(self.W, error, self.deltas)
nervanagpu.bprop_conv(F, E, grad_I):
    layer.bprop_kernels.bind_params(E, F, grad_I)
backends.layer_gpu.ConvLayer():
    self.bprop_kernels = convolution.BpropCuda(lib, self.dtype, N, C, K, D, H, W, T, R, S, M, P, Q,
                                                   pad_d, pad_h, pad_w, str_d, str_h, str_w, bsum=bsum)
convolution.BpropCuda.bind_params(I, F, O):
    kernel(I, filter_temp, O)
kernel(I, F, O)
```

### update gradW

```
layers.layer.bprop():
    be.update_conv(self.inputs, error, self.dW)
nervanagpu.update_conv(I, E, grad_F)
    layer.updat_kernels.bind_params(I, E, grad_F)
convolution.UpdateCuda.bind_params(I, E, O):
    kernel(I, E, O)
kernel(I, F, O)
```

## proposed

### bprop gradI

```
layers.layer.bprop():
    be.brop_conv(error, self.W, self.deltas)
nervanagpu.bprop_conv(gradO, W, gradI):
    layer.bprop_kernels.bind_params(gradO, W, gradI)
backends.layer_gpu.ConvLayer():
    self.bprop_kernels = convolution.BpropCuda(lib, self.dtype, N, C, K, D, H, W, T, R, S, M, P, Q,
                                                   pad_d, pad_h, pad_w, str_d, str_h, str_w, bsum=bsum)
convolution.BpropCuda.bind_params(gradO, W, gradI):
    clRunner.update(gradO, Wt, gradI)
clRunner.update(gradO, Wt, gradI):
    kernel(gradO, Wt, gradI)
kernel(I, F, O)
```

### update gradW

```
layers.layer.bprop():
    be.update_conv(self.inputs, error, self.dW)
nervanagpu.update_conv(I, gradO, gradW)
    layer.updat_kernels.bind_params(I, gradO, gradW)
convolution.UpdateCuda.bind_params(I, gradO, gradW):
    clRunner.update(I, gradO, gradW)
clRunner.update(I, gradO, gradW):
    kernel(I, gradO, gradW)
kernel(I, F, O)
```

