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


def calcU(ctx):
    print('calcU')
    code = r"""
kernel void calcU(
    global float* Out, global const float* In,
    int RSK, int SK, int SK2, int K, int C1152, int C, int GK)
{
    int tid  = get_local_id(0);
    //if(tid != 0) {
    //  return;
    //}
    int blkK = get_num_groups(0) - get_group_id(0) - 1;
    int c    = get_num_groups(1) - get_group_id(1) - 1;
    int k    = (blkK<<5) + tid;

    // output before:
    // [Co//32][Ci][xi][nu][Co % 32]
    
    // output in new order:
    // [xi][nu][Co//32][Ci][Co % 32]
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
    // output in new order:
    // [xi][nu][Co//32][Ci][Co % 32]

    // we can probably make these kernel parameters
    int nu_stride = 32 * C * GK;
    int xi_stride = nu_stride * 6;
    //int nu_stride = 0;
    //int xi_stride = 0;
    out_offset = tid +                // Co % 32
                 (c << 5) +           // Ci
                 ((blkK * C) << 5)    // Co // 32
                 ;
    #pragma unroll
    for (int i = 0; i < 6; i++)
    {
        float t0 = rcp6 * T[i][2];
        float t1 = fma(T[i][0], -rcp6, -t0);
        float t2 = fma(T[i][0], rcp24,  t0);
        // Out[out_offset + 32*(i*6 + 0)] = (rcp4 * T[i][0]);
        // Out[out_offset + 32*(i*6 + 1)] = (fma(T[i][1], -rcp6,  t1));
        // Out[out_offset + 32*(i*6 + 2)] = (fma(T[i][1],  rcp6,  t1));
        // Out[out_offset + 32*(i*6 + 3)] = (fma(T[i][1],  rcp12, t2));
        // Out[out_offset + 32*(i*6 + 4)] = (fma(T[i][1], -rcp12, t2));
        // Out[out_offset + 32*(i*6 + 5)] = (T[i][2]);

        // output in new order:
        // [xi][nu][Co//32][Ci][Co % 32]

        Out[out_offset + i * xi_stride + 0 * nu_stride] = (rcp4 * T[i][0]);
        Out[out_offset + i * xi_stride + 1 * nu_stride] = (fma(T[i][1], -rcp6,  t1));
        Out[out_offset + i * xi_stride + 2 * nu_stride] = (fma(T[i][1],  rcp6,  t1));
        Out[out_offset + i * xi_stride + 3 * nu_stride] = (fma(T[i][1],  rcp12, t2));
        Out[out_offset + i * xi_stride + 4 * nu_stride] = (fma(T[i][1], -rcp12, t2));
        Out[out_offset + i * xi_stride + 5 * nu_stride] = (T[i][2]);
    }

}
"""
    with open('/tmp/out.cl', 'w') as f:
        f.write(code)

    module = cl.Program(ctx, code).build(options='')  # -cl-mad-enable -cl-fast-relaxed-math -cl-no-signed-zeros
    return module.__getattr__('calcU')

def calcV(ctx):
    print('calcV')

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

kernel void calcV(
    global float* Out, global const float* In,
    int Y, int X, int N, int pad_y, int pad_x,
    int GXS, int GYS2, int GXS2, int magic_GXS2, int shift_GXS2, int magic_GXS, int shift_GXS,
    int shlY, int shlX, int maskY, int shrY, int maskX, int shrX, int shlN, int maskN,
    int YXN, int XN, int GYS_GXS_C_1152, int GXS_C_1152, int C_1152,
    int GX, int GY_GX, int GN, int C)
{
    int tid   = get_local_id(0);
    int blkN  = get_num_groups(0) - get_group_id(0) - 1;
    int blkYX = get_num_groups(1) - get_group_id(1) - 1;
    int c     = get_num_groups(2) - get_group_id(2) - 1;

    // unpack y,x from get_group_id(0)
    int gy2 = (blkYX * magic_GXS) >> shift_GXS;
    int gx2 = blkYX - gy2*GXS;
    
    // Implement a square wave block id remapping
    // (for all but last row (if odd number of rows))
    //int gy = gy2 << 1;
    //int gx = gx2;
    //if (gy2 != GYS2)
    //{
    //    gy += (gx2 & 1) ^ ((gx2 & 2) >> 1);
     //   gx  = gx2 >> 1;
    //}
    // Scan backwards on odd rows
    //if (gy2 & 1)
    //    gx = GXS - gx - 1;
        
    int gx = gx2;
    int gy = gy2;

    //int gygx = gy * tiles + gx;

    // Super block YXN coordinates
    int y0 = (gy << shlY) + (((tid & maskY) >> shrY) << 2) - pad_y;
    int x0 = (gx << shlX) + (((tid & maskX) >> shrX) << 2) - pad_x;
    int n  = (blkN << shlN) + (tid & maskN);

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
    // old layout:
    // [tH, tW,           N // 32, Ci, xi, nu,  N % 32]

    // new layout:
    // [xi, nu, N // 32, tH, tW, Ci, N % 32]
    // (note: since last dimension is 32, this is always going to be 128-byte aligned)

    int out_offset = tid +                      // N % 32
                     (c << 5) +                 // ci
                     blkYX * (C << 5) +            // th* tiles + tw (?)
                     // 0 *((2 - gy) * 3 + (2 - gx)) * (C << 5) +            // th* tiles + tw (?)
                     blkN * GY_GX * (C << 5)  //   N // 32
                     ;
    // int out_offset = blkN*GYS_GXS_C_1152 + gy*GXS_C_1152 + gx*C_1152 + c*1152 + tid;

    int nu_stride = GN * GY_GX * (C << 5);
    int xi_stride = nu_stride * 6;

    #pragma unroll
    for (int i = 0; i < 6; i++)
    {
        float t0 = fma(T[i][2], -4.0f, T[i][4]);
        float t1 = fma(T[i][1], -4.0f, T[i][3]);
        float t2 = T[i][4] - T[i][2];
        float t3 = T[i][3] - T[i][1];
        float t4 = fma(T[i][2], -5.0f, T[i][4]);
        float t5 = fma(T[i][3], -5.0f, T[i][5]);

        Out[out_offset + i * xi_stride + 0 * nu_stride] = (fma(T[i][0], 4.0f, t4));
        Out[out_offset + i * xi_stride + 1 * nu_stride] = (t0 + t1);
        Out[out_offset + i * xi_stride + 2 * nu_stride] = (t0 - t1);
        Out[out_offset + i * xi_stride + 3 * nu_stride] = (fma(t3,  2.0f, t2));
        Out[out_offset + i * xi_stride + 4 * nu_stride] = (fma(t3, -2.0f, t2));
        Out[out_offset + i * xi_stride + 5 * nu_stride] = (fma(T[i][1], 4.0f, t5));

        //Out[out_offset + i * xi_stride + 0 * nu_stride] = 123.45f;
    }
    //Out[0] = get_num_groups(1);
    //Out[get_group_id(1) + 1] = (float)gy;
    //Out[get_group_id(1) + 10] = (float)gx;
    //if(get_local_id(0) == 0) {
    //    Out[get_group_id(2)] = get_group_id(2);
    //}
}
"""
    module = cl.Program(ctx, code).build(options='')  # -cl-mad-enable -cl-fast-relaxed-math -cl-no-signed-zeros
    return module.__getattr__('calcV')

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

def calcM(ctx):
    # grid:  (GK, GN, th_tw)
    # block: (32, 1, 1)   # each thread used for different Ci value
    code = r"""
void process_ci_block_too_complicated_do_simple_for_now(
    global float *restrict M, global float *restrict U, global float *restrict V,
        int Ci, int gci, int gk, int gn, int th_tw) {

    // each workgroup handles:
    // 32 values for Co
    // some values for Ci (up to 512?)
    // pull down in blocks of 32 * 4 = 128 floats at a time
    // assume that only hav eup to Ci == 128 for now (add an extra loop later)
    local float4 U4_[32 * 32];
    local float4 V4_[32];
    // int numVRounds = Ci >> (5+2);  // +2 is because we are going to use float4s
    int tid = get_local_id(0);

    int localCi = Ci - (gci << 128);

    int V_offset = 0; // TODO
    float4 V_value = V[V_offset + tid];
    int U_offset = tid;
    global float4 *restrict U4 = (global float4 *)U;
    for(int i = 0; i < 8; i+= 1) {
        // there are: 128 * 32 = 4096 floats
        // or: 32 * 32 = 1024 float4's
        // divided by 32 threads, 32 float4's per thread
        // or 128 floats per thread
        // each loop the 32 threads get 128 float4's, or 512 floats
        // after 8 runs through the loop, it has fetchs 1024 float4's
        int U_offset0 = U_offset + 0;
        int U_offset1 = U_offset + 32;
        int U_offset2 = U_offset + 64;
        int U_offset3 = U_offset + 96;

        float4 b0 = U_offset0 < localCi ? U4[U_offset0] : 0.0f;
        float4 b1 = U_offset0 < localCi ? U4[U_offset1] : 0.0f;
        float4 b2 = U_offset0 < localCi ? U4[U_offset2] : 0.0f;
        float4 b3 = U_offset0 < localCi ? U4[U_offset3] : 0.0f;

        U4_[U_offset0] = b0;
        U4_[U_offset1] = b1;
        U4_[U_offset2] = b2;
        U4_[U_offset3] = b3;

        U_offset += 128;
    }
    V4_[tid] = V_value;
    // no need to sync, since workgroup is 32 threads, equals warpsize (whether this is a good
    // idea, I'm not sure, but worth a shot...)

    // now, all data should have been loaded
    // each thread will sum across all values of ci, for one particular value of co

    local float * restrict U_ = (local float * restrict)U4_;
    local float * restrict V_ = (local float * restrict)V4_;

    float sum = 0;
    for(int ci = 0; ci < Ci; ci += 4) {
        //float s0 = U_[(ci << 5) + tid] * V_
        
        //s0 += s1;
        //s2 += s3;
        //sum = s0 + s2;
    }

}

void process_ci_block(
    global float *restrict M, global float *restrict U, global float *restrict V,
        int Ci, int tiles, int GN, int GK, int b,
        local float *U_, local float *V_) {
    // for now, let's do simple and stupid, no float4s or anything, just get something working :-P
    // also, dont worry about register spills, occupancy etc ...

    int tid = get_local_id(0);
    // stupidly loop over xi and nu for now, to at least get a baseline time, which we can improve a bit...
    int xinu_U_stride = GK * Ci * 32;  // assuming all 32 for now :-P
    int xinu_V_stride = GN * tiles * tiles * Ci * 32;  // assuming 32 again
    int Ci_blocks = (Ci + 31) >> 5;  // blocks of 32 for now, keep it simple
    int tiles_offset = b * Ci * 32;
    for(int gn = 0; gn < GN; gn++) { // loop stpuidly for now...
        int gn32 = gn << 5;
        for(int gk = 0; gk < GK; gk++) {
            int gk32 = gk << 5;
            for(int xi = 0; xi < 6; xi++) {
                for(int nu=0; nu < 6; nu++) {
                    int xinu = xi * 6 + nu;
                    float sum_by_n[32];
                    for(int n = 0; n < 32; n++) {
                        sum_by_n[n] = 0.0f;
                    }
                    // int global_co = gk32 + tid;
                    for(int ci_block = 0; ci_block < Ci_blocks; ci_block++) {
                        // naive again for now...
                        int ci_block_start = ci_block << 5;
                        int local_ci = tid;
                        int local_ci32 = local_ci << 5;
                        int global_ci = ci_block_start + tid;
                        int global_ci32 = global_ci << 5;
                        if(global_ci < Ci) {
                            for(int local_co = 0; local_co < 32; local_co++) {
                                // just copy directly, ignore latency hiding for now
                                U_[local_ci32 + local_co] = U[xinu * xinu_U_stride + gk * Ci * 32 + global_ci32 + local_co];
                            }
                            for(int n = 0; n < 32; n++) {
                                // just copy directly, ignore latency hiding for now
                                V_[local_ci32 + n] = V[xinu * xinu_V_stride + gn * tiles * tiles * Ci * 32 + tiles_offset + global_ci32 + n];
                            }
                            // no need to sync threads, since workgroup size == warpsize, ie 32 (TODO: AMD)
                            
                            // each thread handles ... hmmm...we'd better have 1024 threads (on NVIDIA), or 256 anyway (works also on AMD)
                            // anyway, for now, each thread handles one value of co, all values of n block, and all values of ci
                            // two loops
                        }
                        int local_co = tid;
                        for(int n=0; n < 32; n++) {  // obvioulsy these hould be variables and stuff, in later version
                           float sum = 0.0f;
                           //int global_n = gn32 + n;
                           #pragma unroll
                           for(int ci = 0; ci < 32; ci++) {
                              int global_ci = ci_block_start + ci;  // this is so inefficient...
                              int ci32 = ci << 5;
                              float value = global_ci < Ci ? U_[ci32 + local_co] * V_[ci32 + n] : 0.0f;
                              sum += value;
                           }
                           sum_by_n[n] += sum;
                        }
                    }
                    int local_co = tid;
                    for(int n=0; n < 32; n++) {  // obvioulsy these hould be variables and stuff, in later version
                       // [n//32][n % 32][co // 32][co % 32][th][tw][xi][nu]
                       int offset = (gn32 + n) * GK * 32 * tiles * tiles * 6 * 6 + // (n // 32) * 32 + (n % 32)
                                    (gk32 + local_co) * tiles * tiles * 6 * 6 + // (co % 32)
                                    b * 6 * 6 +   // b
                                    xinu   // xinu
                                    ;
                       M[offset] = sum_by_n[n];
                    }
                }
            }
        }
    }
}

// [n // 32][n % 32][co // 32][co % 32][th][tw][xi][nu]
kernel void calcM(global float *restrict M, const global float *restrict U, const global float *restrict V,
        int Ci, int GCi, int tiles, int GN, int GK,
        local float *U_, local float *V_
    ) {

    int b = get_group_id(0);

    process_ci_block(M, U, V, Ci, tiles, GN, GK, b, U_, V_);
}
    """
    module = cl.Program(ctx, code).build(options='')  # -cl-mad-enable -cl-fast-relaxed-math -cl-no-signed-zeros
    return module.__getattr__('calcM')

def calcO(ctx):
    # grid:  (GK, GN, th_tw)
    # block: (32, 1, 1)   # each thread used for different Ci value
    code = r"""
kernel void calcO(
    global float *O, global float *M, int GID) {
    // lets just do stupidly for now, improve later...
    // assume block (32,1,1)
    // for calcU, each thread does one entire tile (6x6)  Let's do the same thing
    // let's just have a linear grid for now, to keep it simple stupid, then improve it later
    int gid = get_global_id(0);
    if(gid >= GID) {
        return;
    }
    // let's assume this is ... well it doesnt matter actually, we simply do the same operation for all
    // so just grab a tile, and transform it...
    int M_offset = gid * 6 * 6; // 6x6 tiles
    float M_[6][6];
    for(int i = 0; i < 6; i++) {
        int i6 = i * 6;
        #pragma unroll
        for(int j = 0; j < 6; j++) {
            M_[i][j] = M[M_offset + i6 + j];
        }
    }
    float Otmp[4][6];
    for(int i = 0; i < 6; i++) {
        Otmp[0][i] = M_[0][i] + M_[1][i] + M_[2][i] + M_[3][i] + M_[4][i];
        Otmp[1][i] =          + M_[1][i] - M_[2][i] + 2.0f * M_[3][i] - 2.0f * M_[4][i];
        Otmp[2][i] =          + M_[1][i] + M_[2][i] + 4.0f * M_[3][i] + 4.0f * M_[4][i];
        Otmp[3][i] =          + M_[1][i] - M_[2][i] + 8.0f * M_[3][i] - 8.0f * M_[4][i] + M_[5][i];
    }

    global float *restrict O_ = O + gid * 4 * 4;
    for(int i = 0; i < 4; i++) {
        int i4 = (i << 2);
        O_[i4 + 0] = Otmp[i][0] + Otmp[i][1] + Otmp[i][2] + Otmp[i][3] + Otmp[i][4];
        O_[i4 + 1] =         + Otmp[i][1] - Otmp[i][2] + 2.0f * Otmp[i][3] - 2.0f * Otmp[i][4];
        O_[i4 + 2] =         + Otmp[i][1] + Otmp[i][2] + 4.0f * Otmp[i][3] + 4.0f * Otmp[i][4];
        O_[i4 + 3] =         + Otmp[i][1] - Otmp[i][2] + 8.0f * Otmp[i][3] - 8.0f * Otmp[i][4] + Otmp[i][5];
    }
}
    """
    module = cl.Program(ctx, code).build(options='')  # -cl-mad-enable -cl-fast-relaxed-math -cl-no-signed-zeros
    return module.__getattr__('calcO')

