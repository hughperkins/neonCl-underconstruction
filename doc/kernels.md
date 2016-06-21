# Kernel analysis

There is a [long heuristic](https://github.com/NervanaSystems/neon/blob/bc196cbe4131a76cd0c584e93aa7f8285b6243cb/neon/backends/layer_gpu.py#L404-L488) for choosing between kernels.  So which ones are used in practice?

Here, this uses the `analysis` branch, in conjunction with [neon-benchmarks](https://github.com/hughperkins/neon-benchmarks)
to check.

This analysis is done using maxwell kernels (non-maxwell just uses the cuda direct kernels, which are already ported
to opencl).  They were all run on Titan X.

## alexnet

```
0: FpropDirect BpropDirect UpdateDirect  (because kernelsize != 3)
1: FpropDirect BpropDirect UpdateDirect  (because kernelsize != 3)
2: FpropWinograd_4x4_3x3 BpropWinograd_4x4_3x3 UpdateWinograd_3x3_4x4
3: FpropWinograd_4x4_3x3 BpropWinograd_4x4_3x3 UpdateWinograd_3x3_4x4
4: FpropWinograd_4x4_3x3 BpropWinograd_4x4_3x3 UpdateWinograd_3x3_4x4
```

## vgga

```
0: FpropDirect BpropWinograd_4x4_3x3 UpdateDirect
1: FpropWinograd_4x4_3x3 BpropWinograd_4x4_3x3 UpdateWinograd_3x3_4x4
2: FpropWinograd_4x4_3x3 BpropWinograd_4x4_3x3 UpdateWinograd_3x3_4x4
3: FpropWinograd_4x4_3x3 BpropWinograd_4x4_3x3 UpdateWinograd_3x3_4x4
4: FpropWinograd_4x4_3x3 BpropWinograd_4x4_3x3 UpdateWinograd_3x3_4x4
5: FpropWinograd_4x4_3x3 BpropWinograd_4x4_3x3 UpdateWinograd_3x3_4x4
6: FpropWinograd_4x4_3x3 BpropWinograd_4x4_3x3 UpdateWinograd_3x3_4x4
7: FpropWinograd_4x4_3x3 BpropWinograd_4x4_3x3 UpdateWinograd_3x3_4x4
```

## transform kernels

### fprop

```
image_kernel = _get_xprop_image_trans_4x4_kernel(f4)
filter_kernel = _get_fprop_filter_trans_4x4_kernel(f4)
```

### bprop

```
_get_bprop_filter_trans_4x4_kernel(f4)
_get_update_image_trans_4x4_kernel(f4)
_get_update_delta_trans_4x4_kernel(f4)
```

