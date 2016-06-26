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


def fprop_filter_trans_4x4_kernel(ctx):
    print('get_fprop_filter_trans_4x4_kernel')
    code = r"""
kernel void fprop_filter_trans_4x4(
    global float* Out, global const float* In,
    int RSK, int SK, int SK2, int K, int C1152)
{
    int tid  = get_local_id(0);
    //if(tid != 0) {
    //  return;
    //}
    int blkK = get_num_groups(0) - get_group_id(0) - 1;
    int c    = get_num_groups(1) - get_group_id(1) - 1;
    int k    = (blkK<<5) + tid;

    int out_offset = blkK*C1152 + c*1152 + tid;

    bool valid_k = k < K;

    int f_r0s0 = c*RSK  + k;
    int f_r0s1 = f_r0s0 + K;
    int f_r0s2 = f_r0s1 + K;

    int f_r1s0 = f_r0s0 + SK;
    int f_r1s1 = f_r0s1 + SK;
    int f_r1s2 = f_r0s2 + SK;

    int f_r2s0 = f_r0s0 + SK2;
    int f_r2s1 = f_r0s1 + SK2;
    int f_r2s2 = f_r0s2 + SK2;

    float I[3][3];

    I[0][0] = valid_k ? (In[f_r0s0]) : 0.0f;
    I[0][1] = valid_k ? (In[f_r0s1]) : 0.0f;
    I[0][2] = valid_k ? (In[f_r0s2]) : 0.0f;

    I[1][0] = valid_k ? (In[f_r1s0]) : 0.0f;
    I[1][1] = valid_k ? (In[f_r1s1]) : 0.0f;
    I[1][2] = valid_k ? (In[f_r1s2]) : 0.0f;

    I[2][0] = valid_k ? (In[f_r2s0]) : 0.0f;
    I[2][1] = valid_k ? (In[f_r2s1]) : 0.0f;
    I[2][2] = valid_k ? (In[f_r2s2]) : 0.0f;


    float rcp4  = 1.0f/4.0f;
    float rcp6  = 1.0f/6.0f;
    float rcp12 = 1.0f/12.0f;
    float rcp24 = 1.0f/24.0f;
    float T[6][3];
    #pragma unroll
    for (int i = 0; i < 3; i++)
    {
        float t0 = rcp6 * I[2][i];
        float t1 = fma(I[0][i], -rcp6, -t0);
        float t2 = fma(I[0][i], rcp24,  t0);
        T[0][i] = rcp4 * I[0][i];
        T[1][i] = fma(I[1][i], -rcp6,  t1);
        T[2][i] = fma(I[1][i],  rcp6,  t1);
        T[3][i] = fma(I[1][i],  rcp12, t2);
        T[4][i] = fma(I[1][i], -rcp12, t2);
        T[5][i] = I[2][i];
    }
    #pragma unroll
    for (int i = 0; i < 6; i++)
    {
        float t0 = rcp6 * T[i][2];
        float t1 = fma(T[i][0], -rcp6, -t0);
        float t2 = fma(T[i][0], rcp24,  t0);
        Out[out_offset + 32*(i*6 + 0)] = (rcp4 * T[i][0]);
        Out[out_offset + 32*(i*6 + 1)] = (fma(T[i][1], -rcp6,  t1));
        Out[out_offset + 32*(i*6 + 2)] = (fma(T[i][1],  rcp6,  t1));
        Out[out_offset + 32*(i*6 + 3)] = (fma(T[i][1],  rcp12, t2));
        Out[out_offset + 32*(i*6 + 4)] = (fma(T[i][1], -rcp12, t2));
        Out[out_offset + 32*(i*6 + 5)] = (T[i][2]);
    }

}
"""
    with open('/tmp/out.cl', 'w') as f:
        f.write(code)

    module = cl.Program(ctx, code).build(options='')  # -cl-mad-enable -cl-fast-relaxed-math -cl-no-signed-zeros
    return module.__getattr__('fprop_filter_trans_4x4')

def xprop_image_trans_4x4_kernel(ctx):
    print('get_xprop_image_trans_4x4_kernel')

    code = r"""
static inline int div64(int value, int div_mul, int div_shift)
{
    int result;
    // if the divisor is a power of two the magic will be 1 and it's just a simple right shift
    if (div_mul == 1)
        result = value >> div_shift;
    // Otherwise multiply by magic and right shift just the high bits
    else
        result = (value * div_mul) >> div_shift;
    return result;
}

kernel void xprop_image_trans_4x4(
    global float* Out, global const float* In,
    int Y, int X, int N, int pad_y, int pad_x,
    int GXS, int GYS2, int GXS2, int magic_GXS2, int shift_GXS2,
    int shlY, int shlX, int maskY, int shrY, int maskX, int shrX, int shlN, int maskN,
    int YXN, int XN, int GYS_GXS_C_1152, int GXS_C_1152, int C_1152)
{
    int tid   = get_local_id(0);
    int blkN  = get_num_groups(0) - get_group_id(0) - 1;
    int blkYX = get_num_groups(1) - get_group_id(1) - 1;
    int c     = get_num_groups(2) - get_group_id(2) - 1;

    // unpack y,x from get_group_id(0)
    int gy2 = div64(blkYX, magic_GXS2, shift_GXS2);
    int gx2 = blkYX - gy2*GXS2;

    // Implement a square wave block id remapping
    // (for all but last row (if odd number of rows))
    int gy = gy2 << 1;
    int gx = gx2;
    if (gy2 != GYS2)
    {
        gy += (gx2 & 1) ^ ((gx2 & 2) >> 1);
        gx  = gx2 >> 1;
    }
    // Scan backwards on odd rows
    if (gy2 & 1)
        gx = GXS - gx - 1;

    // Super block YXN coordinates
    int y0 = (gy << shlY) + (((tid & maskY) >> shrY) << 2) - pad_y;
    int x0 = (gx << shlX) + (((tid & maskX) >> shrX) << 2) - pad_x;
    int n  = (blkN << shlN) + (tid & maskN);

    int out_offset = blkN*GYS_GXS_C_1152 + gy*GXS_C_1152 + gx*C_1152 + c*1152 + tid;

    bool valid = n < N;

    bool xin[6], yin[6];
    float I[6][6];

    #pragma unroll
    for (int i = 0; i < 6; i++)
    {
        xin[i] = x0 + i >= 0 && x0 + i < X && valid;
        yin[i] = y0 + i >= 0 && y0 + i < Y;
    }

    int offset = c*YXN + y0*XN + x0*N + n;

    #pragma unroll
    for (int y = 0; y < 6; y++)
    {
        if (y) offset += XN;

        #pragma unroll
        for (int x = 0; x < 6; x++)
        {
            float val = 0;
            if (yin[y] && xin[x])
                val = *(In + offset + x*N);
            I[y][x] = (val);
        }
    }

    float T[6][6];
    #pragma unroll
    for (int i = 0; i < 6; i++)
    {
        float t0 = fma(I[2][i], -4.0f, I[4][i]);
        float t1 = fma(I[1][i], -4.0f, I[3][i]);
        float t2 = I[4][i] - I[2][i];
        float t3 = I[3][i] - I[1][i];
        float t4 = fma(I[2][i], -5.0f, I[4][i]);
        float t5 = fma(I[3][i], -5.0f, I[5][i]);
        T[0][i] = fma(I[0][i], 4.0f, t4);
        T[1][i] = t0 + t1;
        T[2][i] = t0 - t1;
        T[3][i] = fma(t3,  2.0f, t2);
        T[4][i] = fma(t3, -2.0f, t2);
        T[5][i] = fma(I[1][i], 4.0f, t5);
    }
    #pragma unroll
    for (int i = 0; i < 6; i++)
    {
        float t0 = fma(T[i][2], -4.0f, T[i][4]);
        float t1 = fma(T[i][1], -4.0f, T[i][3]);
        float t2 = T[i][4] - T[i][2];
        float t3 = T[i][3] - T[i][1];
        float t4 = fma(T[i][2], -5.0f, T[i][4]);
        float t5 = fma(T[i][3], -5.0f, T[i][5]);
        Out[out_offset + 32*(i*6 + 0)] = (fma(T[i][0], 4.0f, t4));
        Out[out_offset + 32*(i*6 + 1)] = (t0 + t1);
        Out[out_offset + 32*(i*6 + 2)] = (t0 - t1);
        Out[out_offset + 32*(i*6 + 3)] = (fma(t3,  2.0f, t2));
        Out[out_offset + 32*(i*6 + 4)] = (fma(t3, -2.0f, t2));
        Out[out_offset + 32*(i*6 + 5)] = (fma(T[i][1], 4.0f, t5));
    }
}
"""
    module = cl.Program(ctx, code).build(options='')  # -cl-mad-enable -cl-fast-relaxed-math -cl-no-signed-zeros
    return module.__getattr__('xprop_image_trans_4x4')

def calcM_blocked_l2(ctx):
    code = r"""
    kernel void calcM_blocked_l2(global float *R, const global float *U, const global float *V,
        int A, int B
        ) {
        // just do really naive for now, improve later...
        // assume block (32,1,1), which fills the warps, ignore shared memory for now
        // incoming data is (A,B).T * (A)  ie (B,A) * (A)
        // result will be (B)
        // B is always 32
        // lets use 1 thread for each B value.
        // first, we should pull down all the data
        // no need for sync, because we are using (32,1,1) block, exactly matches warp
        // then each thread calculates one output value
        // lets do output value first ,since easiest, thne pull down the data
        

        int b = get_local_id(0);
        float sum = 0;
        int A_blocks = A >> 5; // assume A is multipel of 32
        for(int a_block = 0; a_block < A; a_block+= 32) {
            #pragma unroll
            for(int a_local = 0; a_local < 32; a_local++) {
                int a = a_block + a_local;
                // this will be really high latency.  improve later
                sum += U[a<<5 + b] * V[a];
            }
        }
        R[b] = sum;
    }
    """
    module = cl.Program(ctx, code).build(options='')  # -cl-mad-enable -cl-fast-relaxed-math -cl-no-signed-zeros
    return module.__getattr__('calcM_blocked_l2')

def fprop_calcM(ctx):
    # grid:  (GK, GN, th_tw)
    # block: (32, 1, 1)   # each thread used for different Ci value
    code = r"""
void process_ci_block(
    private float sums [6][6],
    global float *restrict M, global float *restrict U, global float *restrict V,
        int gci, int gk, int gn, int th_tw) {
    // load the block first... need somewhere to store it first
    // each thead handles one co values, and all x/y values
    
    // first download the data, into shared memory
    // we have to donwload:
    // U
    // V

    int tid = get_local_id(0);
    // U first
    local float4 U4[6*6*32];
    local float *U_ = (local float *)U4; // block of 4 ci values at a time??? If too much, we lose occupancy...
    int offset = gci << 2 ;
    #pragma unroll
    for(int i = 0; i < 6*6*32; i+= 32) {
        U4[tid + i] = U[offset + tid + i];
    }
    // no need for sync, since num workgroup threads == warpsize
    
    // each thread handles one value of co, for each of the other values
//    for(int ci=0; ci < 4; ci++) {
 //      for(int xi=0; xi < 6; xi++) {
  //         #pragma unroll
   //        for(int nu=0; nu < 6; nu++) {
    //           U_[ci][xi][nu][tid] = U[
     //      }
      // }
    //}

    // now process it
    for(int xi=0; xi < 6; xi++) {
        for(int nu=0; nu < 6; nu++) {
           for(int ci=0; ci < 4; ci++) {
           }
        }
    }
}

kernel void fprop_calcM(global float *restrict M, const global float *restrict U, const global float *restrict V,
        int Ci, int GCi
    ) {
    int gk = get_group_id(0);
    int gn = get_group_id(1);
    int th_tw = get_group_id(2);
    //int th = th_tw >> 3;
    //int tw = th_tw & 0x7;

    private sums[6][6];
    for(int gci = 0; gci < GCi; gci++) {
        process_ci_block(sums, M, U, V, gci, gk, gn, th_tw);
    }
}
    """
    module = cl.Program(ctx, code).build(options='')  # -cl-mad-enable -cl-fast-relaxed-math -cl-no-signed-zeros
    return module.__getattr__('fprop_calcM')

