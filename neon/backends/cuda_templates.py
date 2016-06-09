# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
"""
Templates for cuda kernels:
    _ew_template:           generic template?
    _stage_template:        "loop"
                            "red32"
                            "red"
                            "red_ops"
                            "red_out"
    _fin_template
    _init_rand_func:        Initialize LFSR's
    _init_rand_round_func
    _finish_rand_func
    _common_urand_gen
    _common_frand
    _common_round random:  f4, f2, i4, i2, i1
                  nearest: f2, i4, u4, i2, u2, i1, u1
    _common_fp16_to_fp32:  inline assembly conversion function
    _ew_types:             f4,f2,i4,u4,i2,u2,i1,u1
    _ew_strings:
    _is_finite:            inline assembly test function
    _float_ops:            unary and binary element operations
    _reduction_ops:        sum, max, min, argmax, argmin
"""

_common_round = {

    "random": {

        "f4": r"""
__device__ float fp32_to_fp32_rand(
    float val, unsigned& lfsr0, unsigned& lfsr1, unsigned& lfsr2, float rand_scale,
          unsigned rand_mask)
{
    unsigned urand = urand_gen(lfsr0, lfsr1, lfsr2);

    float ret;
    asm("{\n\t"
        ".reg .f32 exponent, frand, result;\n\t"
        "and.b32 exponent, %1, 0xff800000;\n\t"
        "mul.f32 exponent, exponent, %2;\n\t"
        "cvt.rz.f32.u32 frand, %3;\n\t"
        "fma.rz.f32 result, exponent, frand, %1;\n\t"
        "and.b32 %0, result, %4;\n\t"
        "}" : "=f"(ret) : "f"(val), "f"(rand_scale), "r"(urand), "r"(rand_mask));

    return ret;
}
""",
    },
}


_ew_types = {
    "f4": {
        "type": "float",
        "type4": "float4",
        "cvt": "",
        "cvt_out": "",
    }
}

