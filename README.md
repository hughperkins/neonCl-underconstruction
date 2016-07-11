# NeonCl

A port of nervana neon to OpenCL:
- Kepler direct convolution kernels: done :-)
- Implementation of Winograd 4x4,3x3 in Opencl: in progress

The original neon, documentation etc is at https://github.com/nervanasystems/neon

## Direct convolution kernels

These were ported from the neon CUDA direct convolution kernels.  The performance is almost the same (maybe 10-20% difference, on NVIDIA hardware).

### Installation

#### Prerequisites

* NVIDIA GPU (though should be straightforward to make it work also on AMD etc, but for now should be
NVIDIA)
* CUDA and GPU drivers installed
* OpenCL library installed/available

#### Procedure

```
git clone https://github.com/hughperkins/neonCl-underconstruction neonCl
cd neonCl
bash install.sh
```
Before running any script, do:
```
source env3/bin/activate
```

### Performance

Please see [neon-benchmarks](https://github.com/hughperkins/neon-benchmarks/blob/master/results/vgga_summary.md)

Basically, to within ~10-20% of the speed of the original CUDA direct convolution kernels

### Correctness tests

[neon-benchmarks](https://github.com/hughperkins/neon-benchmarks/blob/master/results/vgga_summary.md)

### Status

Working

### Implementation notes

* currently targeted at NVIDIA devices, using OpenCL 1.2, but allowing use of inline PTX assembler
occasionally, where OpenCL 1.2 doesnt include some functionality
* concretely, the following inline methods need to be implemented for other platforms, eg using OpenCL
methods, or platform-specific assembly, or similar:
  * `__shfl`  (OpenCL 2.0 sub-groups?)
  * `__ballot` (OpenCL 2.0 sub groups?)
  * `__popc` (OpenCL 2.0 `popcnt`?)
  * `__atomicAdd` (OpenCL 2.0 atomics?)

## Winograd convolution

There is no CUDA implementation of Winograd available in neon, only SASS.  For now, what I'm doing is:
- I'm using the image and filter transform kernels, which were in CUDA in neon, but I've modified the layout, probably not in a good way, but I couldnt figure out how to use the existing layout :-)
- wrote an OpenCL kernel from scratch for doing the gemm part, and applying the output transformation

Performance at the moment is atrocious.  So, I'm doing a bunch of experiments at [gpu-experiments](https://github.com/hughperkins/gpu-experiments) to figure out how to improve it a bit. I'm also reading through https://github.com/NervanaSystems/maxas/wiki/SGEMM , which seems to be a mine of useful information.

The kernels are in [winograd_kernels_cl.py](winograd_kernels_cl.py).  To run them, run [winograd_cl.py](winograd_cl.py).  For now, they are under development, and I havent provided any way to run them from eg within a framework.  This will follow, once performance is good enough to be useful.

I also ported them to cuda :-)  [winograd_kernels_cuda.py](winograd_kernels_cuda.py).  Run using [winograd_cuda.py](winograd_cuda.py)

### Optimization notes

Let's consider one specific geometry, which will be:
```
image size: 56x56
kernel size: 3x3
stride: 1x1
padding: 1
input channels: 32
output channels: 32
batch size: 32
winograd geometry: F(4x4,3x3)
```

Current timings on a 940M are:
- calcU 0.23ms
- calcV: 3.14ms
- calcM: 42.84ms
- calcO: 4.50ms

Total: 50.71ms

By comparison, using the same geometry with direct convolution (not even the SASS winograd), takes someting like ~1ms, in total :-P

Clearly there's a bunch of work to do :-)

#### Theoretical limits

Let's think about things like:
- the number of operations involved in this geometry
- the minimum bandwidth required

For reference, the Lavin and Gray paper, http://arxiv.org/abs/1509.09308 , 'algorithm 1'

For calcM:
```
operations required = 6 * 6 * tiles * tiles * N * K * C * 2
                    = 72NKC(HW/16) = 4.5NKCHW
                    = 4.5 * 32 * 32 * 32 * 56 * 56 = 4.62e8
incoming bandwidth required:
   U = 6 * 6 * K * C = 36CK = 36864 floats = 145e3 bytes
   V = 6 * 6 * N * tiles * tiles * C
     = 36 * NC * (HW/16) = 2.25NCHW
     = 2.61e6 floats = 1.05e7 bytes
total incoming = 145e3 + 1.05e7 = 1.06e7 bytes
outgoing bandwidth required:
   M = N * K * tiles * tiles * 6 * 6
     = NK(HW/16) * 36
     = 2.25NKHW
     = 2.36e6 floats
     = 9.44e6 bytes
```
Therefore, just based on bandwidth, if max bandwidth for 940M is 14.40GiB/second, we'd expect execution time to be:
```
time = (1.06e7+9.44e6)/14.40/1024/1024/1024 seconds
     = 0.00130 seconds
     = 1.3 milliseconds
Of which:
   incoming: 0.00069seconds = 0.69milliseconds
   outgoing: 0.00061seconds = 0.61milliseconds
```
maximum theoretical flops (assuming all inputs comes from off-chip, and output goes back to off-chip, during M calculation):
```
gflops = 4.62e8 / 0.0013 / 1e9
       = 355 gigaflops/second
```

## License

Apache 2.0, per original neon

