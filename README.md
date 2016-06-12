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

## Status

Latest news:
- fprop is working!
- backprop gradInputs is working!

Next steps, in no particular order:
- port also backprop weights
- port dim_shuffle (?)
- migrate everything to opencl (ie remove the insane cuda-cpu-cl copying currently involved in running
cl kernels against cuda buffers...)

## License

Apache 2.0, per original neon

