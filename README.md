# What's this branch?

This is a port of nervana neon Kepler convolution kernels to OpenCL.

The original neon, documentation etc is at https://github.com/nervanasystems/neon

## Performance

Preliminary results:

For geometry:
- image_size: 224 x 224
- batch_size: 128
- input_filters: 32
- output_filters: 32

...times for forward propagation, on a Titan X, are:
- nervana neon, Maxwell kernels, CUDA (SOTA): 0.025 seconds
- nervana neon, Kepler kernels, CUDA: 0.036 seconds
- nervana neon, OpenCL (this): 0.041 seconds

More detailed benchmarks in progress at [neon-benchmarks](https://github.com/hughperkins/neon-benchmarks)

## Installation

### Prerequisites

* NVIDIA GPU (though should be straightforward to make it work also on AMD etc, but for now should be
NVIDIA)
* GPU drivers installed
* OpenCL library installed/available

### Procedure

```
git clone https://github.com/hughperkins/winogradCl-underconstruction winogradCl
cd winogradCl
bash install.sh
```
Before running any script, do:
```
source env3/bin/activate
```

## Correctness tests

Test code is at:
- [test_perf.py](test_perf.py)  For timings
- [test_perf1.py](test_perf1.py)  For timings in the OpenCL timings in the Performance section above
- [test_correctness.py](test_correctness.py)   For fprop correctness
- [test_correctness_bprop.py](test_correctness_bprop.py)    For gradInput backprop correctness
- [test_correctness_gradweights.py](test_correctness_gradweights.py)    For gradW backprop correctness

## Status

Working :-)

## Future evolutions

Next up:
- create simple API
- run performance test
- add to cltorch

Medium-term:
- make work on Intel HD GPU (I have one)
- make work on AMD GPU (either someone else can do this, or I can do it, if someone can provide me with a cloud-hosted one)

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

