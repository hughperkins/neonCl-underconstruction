# What's this branch?

This fork is a stripped down version of neon, with only the convolution layer, no other layers
or ... anything really.  Just pure convolution.  Also, no sass.  Just Kepler convolutions.

The original neon, documentation etc is at https://github.com/nervanasystems/neon

## How to run something with it?

Test code is at:
- [test_perf.py](test_perf.py)
- [test_correctness.py](test_correctness.py)

## Plan

It's going to contain a version of neon convolutions ported to OpenCL

This is started at:

https://github.com/hughperkins/winogradCl-underconstruction/blob/a6160f0fce407136e178df32417958a8ba77072a/neon/backends/kernels/cl/convolution_cl.py

## License

Apache 2.0, per original neon

