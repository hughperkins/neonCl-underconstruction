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


# from https://github.com/nervanasystems/neon.git file neon/backends/winograd_conv.py

def _get_fprop_filter_trans_kernel(dtype):
    print('_get_fprop_filter_trans_kernel(%s)' % dtype)

    code = r"""
%(common)s

__global__ void fprop_filter_trans(%(type4)s* T, const %(type)s* F, int RSK, int SK, int SK2, int K)
{
    extern %(type)s  __shared__ share[];
    extern %(type4)s __shared__ share4[];

    int tid  = threadIdx.x;
    int blkK = blockIdx.x;
    int c    = blockIdx.y;
    int k    = (blkK<<5) + tid;

    bool valid_k = k < K;

    int f_r0s0 = c*RSK  + k;
    int f_r0s1 = f_r0s0 + K;
    int f_r0s2 = f_r0s1 + K;

    int f_r2s0 = f_r0s0 + SK2;
    int f_r2s1 = f_r0s1 + SK2;
    int f_r2s2 = f_r0s2 + SK2;

    int f_r1s0 = f_r0s0 + SK;
    int f_r1s1 = f_r0s1 + SK;
    int f_r1s2 = f_r0s2 + SK;

    float r0s0 = valid_k ? %(cvt_in)s(__ldg(F + f_r0s0)) : 0.0f;
    float r0s1 = valid_k ? %(cvt_in)s(__ldg(F + f_r0s1)) : 0.0f;
    float r0s2 = valid_k ? %(cvt_in)s(__ldg(F + f_r0s2)) : 0.0f;

    float r2s0 = valid_k ? %(cvt_in)s(__ldg(F + f_r2s0)) : 0.0f;
    float r2s1 = valid_k ? %(cvt_in)s(__ldg(F + f_r2s1)) : 0.0f;
    float r2s2 = valid_k ? %(cvt_in)s(__ldg(F + f_r2s2)) : 0.0f;

    float r1s0 = valid_k ? %(cvt_in)s(__ldg(F + f_r1s0)) : 0.0f;
    float r1s1 = valid_k ? %(cvt_in)s(__ldg(F + f_r1s1)) : 0.0f;
    float r1s2 = valid_k ? %(cvt_in)s(__ldg(F + f_r1s2)) : 0.0f;

    float temp00 = __fmul_rn(r0s1, 0.5f);
    float temp01 = __fadd_rn(r0s0, r0s2);
    float F01    = __fmaf_rn(temp01, 0.5f,  temp00);
    float F02    = __fmaf_rn(temp01, 0.5f, -temp00);
    share[tid + 32*0] = %(cvt_out)s(r0s0);
    share[tid + 32*1] = %(cvt_out)s(F01);
    share[tid + 32*2] = %(cvt_out)s(F02);
    share[tid + 32*3] = %(cvt_out)s(r0s2);
    float temp02 = __fadd_rn(r2s0, r2s2);
    float temp08 = __fmul_rn(r2s1, 0.5f);
    float F13    = __fmaf_rn(temp02, 0.5f,  temp08);
    float F14    = __fmaf_rn(temp02, 0.5f, -temp08);
    share[tid + 32*12] = %(cvt_out)s(r2s0);
    share[tid + 32*13] = %(cvt_out)s(F13);
    share[tid + 32*14] = %(cvt_out)s(F14);
    share[tid + 32*15] = %(cvt_out)s(r2s2);
    float temp10 = __fadd_rn(temp01, temp02);
    float temp05 = __fadd_rn(r0s1,   r2s1);
    float temp07 = __fadd_rn(r1s0,   r1s2);
    float temp09 = __fmul_rn(r1s1,   0.25f);
    float temp11 = __fadd_rn(temp10,  temp05);
    float temp14 = __fadd_rn(temp10, -temp05);
    float temp13 = __fmaf_rn(temp07, 0.25f,  temp09);
    float temp15 = __fmaf_rn(temp07, 0.25f, -temp09);
    float F05    = __fmaf_rn(temp11, 0.25f,  temp13);
    float F09    = __fmaf_rn(temp11, 0.25f, -temp13);
    float F06    = __fmaf_rn(temp14, 0.25f,  temp15);
    float F10    = __fmaf_rn(temp14, 0.25f, -temp15);
    share[tid + 32* 5] = %(cvt_out)s(F05);
    share[tid + 32* 9] = %(cvt_out)s(F09);
    share[tid + 32* 6] = %(cvt_out)s(F06);
    share[tid + 32*10] = %(cvt_out)s(F10);
    float temp03 = __fmul_rn(r1s0, 0.5f);
    float temp06 = __fadd_rn(r0s0, r2s0);
    float temp04 = __fmul_rn(r1s2, 0.5f);
    float F04    = __fmaf_rn(temp06, 0.5f,  temp03);
    float F08    = __fmaf_rn(temp06, 0.5f, -temp03);
    share[tid + 32*4] = %(cvt_out)s(F04);
    share[tid + 32*8] = %(cvt_out)s(F08);
    float temp12 = __fadd_rn(r0s2, r2s2);
    float F07    = __fmaf_rn(temp12, 0.5f,  temp04);
    float F11    = __fmaf_rn(temp12, 0.5f, -temp04);
    share[tid + 32* 7] = %(cvt_out)s(F07);
    share[tid + 32*11] = %(cvt_out)s(F11);

    %(type4)s batch0 = share4[tid +  0];
    %(type4)s batch1 = share4[tid + 32];
    %(type4)s batch2 = share4[tid + 64];
    %(type4)s batch3 = share4[tid + 96];

    int offset = c*gridDim.x*128 + blkK*128 + tid;

    T[offset +  0] = batch0;
    T[offset + 32] = batch1;
    T[offset + 64] = batch2;
    T[offset + 96] = batch3;
}
"""
    common  = _common_round["nearest"].get(dtype, "")
    if dtype == "f2":
        common += _common_fp16_to_fp32

    code = code % {
        "common"  : common,
        "type"    : _ew_types[dtype]["type"],
        "type4"   : _ew_types[dtype]["type4"],
        "cvt_in"  : _ew_types[dtype]["cvt"],
        "cvt_out" : _ew_types[dtype]["cvt_out"],
    }

    module = SourceModule(code)
    kernel = module.get_function("fprop_filter_trans")
    kernel.prepare("PPIIII")
    return kernel

