# Copyright 2014 Nervana Systems Inc., 2016 Hugh Perkins All rights reserved.
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

# Ported to OpenCL from https://github.com/nervanasystems/neon.git file neon/backends/winograd_conv.py

import pyopencl as cl


def get_fprop_filter_trans_kernel(ctx):
    print('get_fprop_filter_trans_kernel')

    code = r"""
kernel void fprop_filter_trans(global float4* T, global const float* F, int RSK, int SK, int SK2, int K,
    local float *share)
{

    local float4 *share4 = (local float4 *)share;

    int tid  = get_local_id(0);
    int blkK = get_group_id(0);
    int c    = get_group_id(1);
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

    float r0s0 = valid_k ? (F[f_r0s0]) : 0.0f;
    float r0s1 = valid_k ? (F[f_r0s1]) : 0.0f;
    float r0s2 = valid_k ? (F[f_r0s2]) : 0.0f;

    float r2s0 = valid_k ? (F[f_r2s0]) : 0.0f;
    float r2s1 = valid_k ? (F[f_r2s1]) : 0.0f;
    float r2s2 = valid_k ? (F[f_r2s2]) : 0.0f;

    float r1s0 = valid_k ? (F[f_r1s0]) : 0.0f;
    float r1s1 = valid_k ? (F[f_r1s1]) : 0.0f;
    float r1s2 = valid_k ? (F[f_r1s2]) : 0.0f;

    float temp00 = r0s1 * 0.5f;
    float temp01 = r0s0 + r0s2;
    float F01    = fma(temp01, 0.5f,  temp00);
    float F02    = fma(temp01, 0.5f, -temp00);
    share[tid + 32*0] = (r0s0);
    share[tid + 32*1] = (F01);
    share[tid + 32*2] = (F02);
    share[tid + 32*3] = (r0s2);
    float temp02 = r2s0 + r2s2;
    float temp08 = r2s1 + 0.5f;
    float F13    = fma(temp02, 0.5f,  temp08);
    float F14    = fma(temp02, 0.5f, -temp08);
    share[tid + 32*12] = (r2s0);
    share[tid + 32*13] = (F13);
    share[tid + 32*14] = (F14);
    share[tid + 32*15] = (r2s2);
    float temp10 = temp01 + temp02;
    float temp05 = r0s1 + r2s1;
    float temp07 = r1s0 + r1s2;
    float temp09 = r1s1 * 0.25f;
    float temp11 = temp10 + temp05;
    float temp14 = temp10 - temp05;
    float temp13 = fma(temp07, 0.25f,  temp09);
    float temp15 = fma(temp07, 0.25f, -temp09);
    float F05    = fma(temp11, 0.25f,  temp13);
    float F09    = fma(temp11, 0.25f, -temp13);
    float F06    = fma(temp14, 0.25f,  temp15);
    float F10    = fma(temp14, 0.25f, -temp15);
    share[tid + 32* 5] = (F05);
    share[tid + 32* 9] = (F09);
    share[tid + 32* 6] = (F06);
    share[tid + 32*10] = (F10);
    float temp03 = r1s0 * 0.5f;
    float temp06 = r0s0 + r2s0;
    float temp04 = r1s2 * 0.5f;
    float F04    = fma(temp06, 0.5f,  temp03);
    float F08    = fma(temp06, 0.5f, -temp03);
    share[tid + 32*4] = (F04);
    share[tid + 32*8] = (F08);
    float temp12 = r0s2 + r2s2;
    float F07    = fma(temp12, 0.5f,  temp04);
    float F11    = fma(temp12, 0.5f, -temp04);
    share[tid + 32* 7] = (F07);
    share[tid + 32*11] = (F11);

    float4 batch0 = share4[tid +  0];
    float4 batch1 = share4[tid + 32];
    float4 batch2 = share4[tid + 64];
    float4 batch3 = share4[tid + 96];

    int offset = c*get_num_groups(0)*128 + blkK*128 + tid;

    T[offset +  0] = batch0;
    T[offset + 32] = batch1;
    T[offset + 64] = batch2;
    T[offset + 96] = batch3;
}
"""
    with open('/tmp/out.cl', 'w') as f:
        f.write(code)

    module = cl.Program(ctx, code).build(options='')  # -cl-mad-enable -cl-fast-relaxed-math -cl-no-signed-zeros
    return module.__getattr__('fprop_filter_trans')

