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

# RAND_POOL_SIZE set to 65536 == 2048 * 32

_ew_template = r"""

#define FLT_MAX 3.402823466E+38F
#define RAND_POOL_SIZE 65536

%(common)s

#define THREADS %(threads)s

__global__ void %(name)s (
    unsigned* rand_state,
    %(arguments)s)
{
    const int tid  = threadIdx.x;
    const int bid  = blockIdx.x;

    extern __shared__ float sPartials[];

    %(inits)s
"""


_fin_template = r"""
    %(finish)s
}
"""


_common_kepler = r"""
#define __ldg(x) (*(x))
"""

_common_urand_gen = r"""
__device__ unsigned urand_gen(unsigned& lfsr0, unsigned& lfsr1, unsigned& lfsr2)
{
    lfsr0 = ((lfsr0 & 0xfffffffe) << 12) ^ (((lfsr0 << 13) ^ lfsr0) >> 19);
    lfsr1 = ((lfsr1 & 0xfffffff8) <<  4) ^ (((lfsr1 << 2)  ^ lfsr1) >> 25);
    lfsr2 = ((lfsr2 & 0xfffffff0) << 11) ^ (((lfsr2 << 3)  ^ lfsr2) >> 11);
    return lfsr0 ^ lfsr1 ^ lfsr2;
}
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
        "f2": r"""
__device__ unsigned short fp32_to_fp16_rand(
    float val, unsigned& lfsr0, unsigned& lfsr1, unsigned& lfsr2, float rand_scale,
          unsigned rand_mask)
{
    unsigned urand = urand_gen(lfsr0, lfsr1, lfsr2);

    unsigned short half;
    asm("{\n\t"
        ".reg .f16 result16;\n\t"
        ".reg .f32 exponent, frand, result32;\n\t"
        "and.b32 exponent, %1, 0xff800000;\n\t"
        "mul.f32 exponent, exponent, %2;\n\t"
        "cvt.rz.f32.u32 frand, %3;\n\t"
        "fma.rz.f32 result32, exponent, frand, %1;\n\t"
        "and.b32 result32, result32, %4;\n\t"
        "cvt.rz.f16.f32 result16, result32;\n\t"
        "mov.b16 %0, result16;\n\t"
        "}" : "=h"(half) : "f"(val), "f"(rand_scale), "r"(urand), "r"(rand_mask));

    return half;
}
""",
        "i4": r"""
__device__ __forceinline__ int fp32_to_int32_rand(
    float val, unsigned& lfsr0, unsigned& lfsr1, unsigned& lfsr2)
{
    unsigned urand = urand_gen(lfsr0, lfsr1, lfsr2);
    int ret;
    asm("{\n\t"
        ".reg .f32 frand, result32;\n\t"
        "cvt.rz.f32.u32 frand, %2;\n\t"
        "copysign.f32 frand, %1, frand;\n\t"
        "mul.rz.f32 frand, frand, 0F2f800000;\n\t"
        "add.rz.f32 result32, frand, %1;\n\t"
        "cvt.rzi.s32.f32 %0, result32;\n\t"
        "}" : "=r"(ret) : "f"(val), "r"(urand));
    return ret;
}
""",
        "i2": r"""
__device__ __forceinline__ short fp32_to_int16_rand(
    float val, unsigned& lfsr0, unsigned& lfsr1, unsigned& lfsr2)
{
    unsigned urand = urand_gen(lfsr0, lfsr1, lfsr2);
    short half;
    asm("{\n\t"
        ".reg .f32 frand, result32;\n\t"
        "cvt.rz.f32.u32 frand, %2;\n\t"
        "copysign.f32 frand, %1, frand;\n\t"
        "mul.rz.f32 frand, frand, 0F2f800000;\n\t"
        "add.rz.f32 result32, frand, %1;\n\t"
        "cvt.rzi.s16.f32 %0, result32;\n\t"
        "}" : "=h"(half) : "f"(val), "r"(urand));
    return half;
}
""",
        "i1": r"""
__device__ __forceinline__ char fp32_to_int8_rand(
    float val, unsigned& lfsr0, unsigned& lfsr1, unsigned& lfsr2)
{
    unsigned urand = urand_gen(lfsr0, lfsr1, lfsr2);
    int ret;
    asm("{\n\t"
        ".reg .f32 frand, result32;\n\t"
        "cvt.rz.f32.u32 frand, %2;\n\t"
        "copysign.f32 frand, %1, frand;\n\t"
        "mul.rz.f32 frand, frand, 0F2f800000;\n\t"
        "add.rz.f32 result32, frand, %1;\n\t"
        "cvt.rzi.s8.f32 %0, result32;\n\t"
        "}" : "=r"(ret) : "f"(val), "r"(urand));
    return ret;
}
""",
    },
    "nearest": {

        "f2": r"""
__device__ __forceinline__ unsigned short fp32_to_fp16(float val)
{
    unsigned short ret;
    asm("{\n\t"
        ".reg .f16 f16;\n\t"
        "cvt.rn.f16.f32 f16, %1;"
        "mov.b16 %0, f16;\n\t"
        "}" : "=h"(ret) : "f"(val));
    return ret;
}
""",
        "i4": r"""
__device__ __forceinline__ int fp32_to_int32(float val)
{
    int ret;
    asm("cvt.rni.s32.f32 %0, %1;" : "=r"(ret) : "f"(val));
    return ret;
}
""",
        "u4": r"""
__device__ __forceinline__ unsigned fp32_to_uint32(float val)
{
    unsigned ret;
    asm("cvt.rni.u32.f32 %0, %1;" : "=r"(ret) : "f"(val));
    return ret;
}
""",
        "i2": r"""
__device__ __forceinline__ short fp32_to_int16(float val)
{
    short ret;
    asm("cvt.rni.s16.f32 %0, %1;" : "=h"(ret) : "f"(val));
    return ret;
}
""",
        "u2": r"""
__device__ __forceinline__ unsigned short fp32_to_uint16(float val)
{
    unsigned short ret;
    asm("cvt.rni.u16.f32 %0, %1;" : "=h"(ret) : "f"(val));
    return ret;
}
""",
        "i1": r"""
__device__ __forceinline__ char fp32_to_int8(float val)
{
    int ret;
    asm("cvt.rni.s8.f32 %0, %1;" : "=r"(ret) : "f"(val));
    return ret;
}
""",
        "u1": r"""
__device__ __forceinline__ unsigned char fp32_to_uint8(float val)
{
    unsigned ret;
    asm("cvt.rni.u8.f32 %0, %1;" : "=r"(ret) : "f"(val));
    return ret;
}
""",
    },
}
# random rounding not yet used for these types
for dtype in ("u4", "u2", "u1"):
    _common_round["random"][dtype] = _common_round["nearest"][dtype]

for mode in ("random", "nearest"):
    for xtype, itype in zip(("x4", "x2", "x1"), ("i4", "i2", "i1")):
        _common_round[mode][xtype] = _common_round[mode][itype]


_common_fp16_to_fp32 = r"""
__device__ __forceinline__ float fp16_to_fp32(unsigned short val)
{
    float ret;
    asm("{\n\t"
        ".reg .f16 f16;\n\t"
        "mov.b16 f16, %1;\n\t"
        "cvt.f32.f16 %0, f16\n\t;"
        "}" : "=f"(ret) : "h"(val));
    return ret;
}
"""

_ew_types = {
    "f4": {
        "type": "float",
        "type4": "float4",
        "cvt": "",
        "cvt_out": "",
    },
    "f2": {
        "type": "unsigned short",
        "type4": "ushort4",
        "cvt": "fp16_to_fp32",
        "cvt_out": "fp32_to_fp16",
    }
}


_ew_strings = {

    # 0: arg_id, 1: stage, 2: type, 3: cvt
    "in0": {
        "arguments": "const {2}* a{0}_in, int row_strd{0}, int col_strd{0}",
        "inits": "const {2}* a{0}_in{1} = a{0}_in + bid * row_strd{0} + tid * col_strd{0};\n"
        "    int a{0}_inc{1} = THREADS * col_strd{0};",
        "loads": "float a{0} = {3}(__ldg(a{0}_in{1}));\n"
        "        a{0}_in{1} += a{0}_inc{1};",
    },
    "in1": {
        "arguments": "const {2}* a{0}_in, int row_strd{0}, int col_strd{0}, const int* take{0}_in",
        "inits": """const {2}* a{0}_in{1} = a{0}_in + __ldg(take{0}_in + bid) * row_strd{0}
                                            + tid * col_strd{0};\n"""
        "    int a{0}_inc{1} = THREADS * col_strd{0};",
        "loads": "float a{0} = {3}(__ldg(a{0}_in{1}));\n"
        "        a{0}_in{1} += a{0}_inc{1};",
    },
    "in2": {
        "arguments": "const {2}* a{0}_in, int row_strd{0}, int col_strd{0}, const int* take{0}_in",
        "inits": "const {2}* a{0}_in{1} = a{0}_in + bid * row_strd{0};\n"
        "    const int* take{0}_in{1} = take{0}_in + tid;",
        "loads": "float a{0} = {3}(__ldg(a{0}_in{1} + __ldg(take{0}_in{1})));\n"
        "        take{0}_in{1} += THREADS;",
    },
    "out0": {
        "arguments": "{2}* a_out, int row_strd, int col_strd",
        "inits": "a_out += bid * row_strd + tid * col_strd;\n"
        "    int out_inc = THREADS * col_strd;",
        "output": "*a_out = {0};\n        a_out += out_inc;",
    },
    "out1": {
        "arguments": "{2}* a_out, int row_strd, int col_strd, const int* take_out",
        "inits": "a_out += __ldg(take_out + bid) * row_strd + tid * col_strd;\n"
        "    int out_inc = THREADS * col_strd;",
        "output": "*a_out = {0};\n        a_out += out_inc;",

    },
    "out2": {
        "arguments": "{2}* a_out, int row_strd, int col_strd, const int* take_out",
        "inits": "a_out += bid * row_strd;\n"
        "    take_out += tid;",

        "output": "*(a_out + __ldg(take_out)) = {0};\n        take_out += THREADS;",

    },
    "const": {
        "arguments": "float c{0}",
    },
}


_is_finite = r"""
float {0};
asm("{{\n\t"
    ".reg .pred is_finite;\n\t"
    "testp.finite.f32 is_finite, %1;\n\t"
    "selp.f32 %0, 0F3f800000, 0F00000000, is_finite;\n\t"
    "}}" : "=f"({0}) : "f"({1}));
"""

