# WinogradCl

A port of nervana neon Kepler direct convolution kernels to OpenCL.

The original neon, documentation etc is at https://github.com/nervanasystems/neon

Note that Kepler kernels are not the fast SASS Winograd ones, but they're fairly ok :-)

## Installation

### Prerequisites

* NVIDIA GPU (though should be straightforward to make it work also on AMD etc, but for now should be
NVIDIA)
* CUDA and GPU drivers installed
* OpenCL library installed/available

### Procedure

```
git clone https://github.com/hughperkins/neonCl-underconstruction neonCl
cd neonCl
bash install.sh
```
Before running any script, do:
```
source env3/bin/activate
```

## Performance

Please see [neon-benchmarks](https://github.com/hughperkins/neon-benchmarks/blob/master/results/vgga_summary.md)

## Correctness tests

[neon-benchmarks](https://github.com/hughperkins/neon-benchmarks/blob/master/results/vgga_summary.md)

## Status

Under construction

## Implementation notes

* currently targeted at NVIDIA devices, using OpenCL 1.2, but allowing use of inline PTX assembler
occasionally, where OpenCL 1.2 doesnt include some functionality
* concretely, the following inline methods need to be implemented for other platforms, eg using OpenCL
methods, or platform-specific assembly, or similar:
  * `__shfl`  (OpenCL 2.0 sub-groups?)
  * `__ballot` (OpenCL 2.0 sub groups?)
  * `__popc` (OpenCL 2.0 `popcnt`?)
  * `__atomicAdd` (OpenCL 2.0 atomics?)

## License

Apache 2.0, per original neon

