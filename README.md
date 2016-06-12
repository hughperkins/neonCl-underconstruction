# What's this branch?

This is a port in progress of nervana neon Kepler convolution kernels to OpenCL.

The original neon, documentation etc is at https://github.com/nervanasystems/neon

## How to run something with it?

Test code is at:
- [test_perf.py](test_perf.py)  For timings
- [test_correctness.py](test_correctness.py)   For fprop correctness
- [test_correctness_bprop.py](test_correctness_bprop.py)    For gradInput backprop correctness
- [test_correctness_gradweights.py](test_correctness_gradweights.py)    For gradW backprop correctness

## Status

Latest news:
- fprop is working!
- backprop gradInputs is working!
- dimshuffle ported to opencl
- backprop gradWeights works now
- pretty much all cuda kernels now removed :-)
- all inputs are now opencl buffers :-)

Next steps, in no particular order:
- create api method to do the convolutions (or three, ie forward, gradInput, gradWeights)
- run some performance benchmarks

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

