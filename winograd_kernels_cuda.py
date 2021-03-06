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

# this is a cuda port of an opencl implementation of Lavin and Gray's winograd algorithms
# idea is that since my gpu is nvidia, it will be easier to optimize using cuda, then backport back
# into opencl???


import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import os
from os import path


if not path.isdir('/tmp/cudaptx'):
    os.makedirs('/tmp/cudaptx')

def calcU():
    print('calcU')
    code = r"""
__global__ void calcU(
    float* Out, const float* In,
    int RSK, int SK, int SK2, int K, int C1152, int C, int GK)
{
    int tid  = threadIdx.x;
    //if(tid != 0) {
    //  return;
    //}
    int blkK = gridDim.x - blockIdx.x - 1;
    int c    = gridDim.y - blockIdx.y - 1;
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

    // we can probably make these __global__ parameters
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
    with open('/tmp/out.cu', 'w') as f:
        f.write(code)

    module = SourceModule(code, keep=True, cache_dir='/tmp/cudaptx')  # -cl-mad-enable -cl-fast-relaxed-math -cl-no-signed-zeros
    return module.get_function('calcU')

def calcV():
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

__global__ void calcV(
    float* Out, const float* In,
    int Y, int X, int N, int pad_y, int pad_x,
    int GXS, int GYS2, int GXS2, int magic_GXS2, int shift_GXS2, int magic_GXS, int shift_GXS,
    int shlY, int shlX, int maskY, int shrY, int maskX, int shrX, int shlN, int maskN,
    int YXN, int XN, int GYS_GXS_C_1152, int GXS_C_1152, int C_1152,
    int GX, int GY_GX, int GN, int C)
{
    int tid   = threadIdx.x;
    int blkN  = gridDim.x - blockIdx.x - 1;
    int blkYX = gridDim.y - blockIdx.y - 1;
    int c     = gridDim.z - blockIdx.z - 1;

    // unpack y,x from blockIdx.x
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
    //Out[0] = gridDim.y;
    //Out[blockIdx.y + 1] = (float)gy;
    //Out[blockIdx.y + 10] = (float)gx;
    //if(threadIdx.x == 0) {
    //    Out[blockIdx.z] = blockIdx.z;
    //}
}
"""
    module = SourceModule(code, keep=True, cache_dir='/tmp/cudaptx')  # -cl-mad-enable -cl-fast-relaxed-math -cl-no-signed-zeros
    return module.get_function('calcV')

def calcM_blocked_l2():
    code = r"""
    __global__ void calcM_blocked_l2(float *R, const float *U, const float *V,
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
        

        int b = threadIdx.x;
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
    module = SourceModule(code, keep=True, cache_dir='/tmp/cudaptx')  # -cl-mad-enable -cl-fast-relaxed-math -cl-no-signed-zeros
    return module.get_function('calcM_blocked_l2')

def calcM():
    # grid:  (GK, GN, th_tw)
    # block: (32, 1, 1)   # each thread used for different Ci value
    code = r"""
__device__ void process_ci_block(
    float *M, const float *U, const float *V,
        int Ci, int tiles, int GN, int GK, int b) {

    __shared__ float U_[32*32];
    __shared__ float V_[32*32];
    int tid1 = threadIdx.y;
    int tid = threadIdx.x;
    int xinu_U_stride = GK * Ci * 32;
    int xinu_V_stride = GN * tiles * tiles * Ci * 32;
    int Ci_blocks = (Ci + 31) >> 5;
    int tiles_offset = b * Ci * 32;
    for(int gn = 0; gn < GN; gn++) {
        int gn32 = gn << 5;
        for(int gk = 0; gk < GK; gk++) {
            int gk32 = gk << 5;
            for(int xi = 0; xi < 6; xi++) {
                for(int nu=0; nu < 6; nu++) {
                    int xinu = xi * 6 + nu;
                    float sum0 = 0.0f;
                    float sum1 = 0.0f;
                    for(int ci_block = 0; ci_block < Ci_blocks; ci_block++) {
                        int ci_block_start = ci_block << 5;
                        int local_ci = tid;
                        int local_ci32 = local_ci << 5;
                        int global_ci = ci_block_start + tid;
                        int global_ci32 = global_ci << 5;
                        __syncthreads();
                        if(global_ci < Ci) {
                            {
                                int local_co = tid1;
                                U_[local_ci32 + local_co] = U[xinu * xinu_U_stride + gk * Ci * 32 + global_ci32 + local_co];
                                U_[local_ci32 + local_co + 16] = U[xinu * xinu_U_stride + gk * Ci * 32 + global_ci32 + local_co + 16];
                            }
                            {
                                int n = tid1;
                                V_[local_ci32 + n] = V[xinu * xinu_V_stride + gn * tiles * tiles * Ci * 32 + tiles_offset + global_ci32 + n];
                                V_[local_ci32 + n + 16] = V[xinu * xinu_V_stride + gn * tiles * tiles * Ci * 32 + tiles_offset + global_ci32 + n + 16];
                            }
                        }
                        __syncthreads();
                        int local_co = tid;
                        {
                          int n = tid1;
                           #pragma unroll
                           for(int ci = 0; ci < 32; ci++) {
                              int global_ci = ci_block_start + ci;
                              int ci32 = ci << 5;
                              float value = global_ci < Ci ? U_[ci32 + local_co] * V_[ci32 + n] : 0.0f;
                              sum0 += value;
                           }
                          n = tid1 + 16;
                           #pragma unroll
                           for(int ci = 0; ci < 32; ci++) {
                              int global_ci = ci_block_start + ci;
                              int ci32 = ci << 5;
                              float value = global_ci < Ci ? U_[ci32 + local_co] * V_[ci32 + n] : 0.0f;
                              sum1 += value;
                           }
                        }
                    }
                    int local_co = tid;
                    {
                       int n = tid1;
                       int offset = (gn32 + n) * GK * 32 * tiles * tiles * 6 * 6 + // (n // 32) * 32 + (n % 32)
                                    (gk32 + local_co) * tiles * tiles * 6 * 6 + // (co % 32)
                                    b * 6 * 6 +   // b
                                    xinu   // xinu
                                    ;
                       M[offset] = sum0;

                       n = tid1 + 16;
                       offset = (gn32 + n) * GK * 32 * tiles * tiles * 6 * 6 + // (n // 32) * 32 + (n % 32)
                                    (gk32 + local_co) * tiles * tiles * 6 * 6 + // (co % 32)
                                    b * 6 * 6 +   // b
                                    xinu   // xinu
                                    ;
                       M[offset] = sum1;
                    }
                }
            }
        }
    }
}

// [n // 32][n % 32][co // 32][co % 32][th][tw][xi][nu]
__global__ void calcM(float *M, const float *U, const float *V,
        int Ci, int GCi, int tiles, int GN, int GK
    ) {

    int b = blockIdx.x;
    //int tid1 = threadIdx.x;

   // if(tid1 == 0) {  // experiment to see if this affects the time
        process_ci_block(M, U, V, Ci, tiles, GN, GK, b);
    //}
}
    """
    with open('/tmp/cudaptx/out.cu', 'w') as f:
        f.write(code)
    
    module = SourceModule(code, keep=True, cache_dir='/tmp/cudaptx')  # -cl-mad-enable -cl-fast-relaxed-math -cl-no-signed-zeros
    return module.get_function('calcM')

def calcO():
    # grid:  (GK, GN, th_tw)
    # block: (32, 1, 1)   # each thread used for different Ci value
    code = r"""
__global__ void calcO(
    float *O, float *M, int GID) {
    // lets just do stupidly for now, improve later...
    // assume block (32,1,1)
    // for calcU, each thread does one entire tile (6x6)  Let's do the same thing
    // let's just have a linear grid for now, to keep it simple stupid, then improve it later
    int gid = (blockIdx.x * blockDim.x + threadIdx.x);
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

    float *O_ = O + gid * 4 * 4;
    for(int i = 0; i < 4; i++) {
        int i4 = (i << 2);
        O_[i4 + 0] = Otmp[i][0] + Otmp[i][1] + Otmp[i][2] + Otmp[i][3] + Otmp[i][4];
        O_[i4 + 1] =         + Otmp[i][1] - Otmp[i][2] + 2.0f * Otmp[i][3] - 2.0f * Otmp[i][4];
        O_[i4 + 2] =         + Otmp[i][1] + Otmp[i][2] + 4.0f * Otmp[i][3] + 4.0f * Otmp[i][4];
        O_[i4 + 3] =         + Otmp[i][1] - Otmp[i][2] + 8.0f * Otmp[i][3] - 8.0f * Otmp[i][4] + Otmp[i][5];
    }
}
    """
    module = SourceModule(code, keep=True, cache_dir='/tmp/cudaptx')  # -cl-mad-enable -cl-fast-relaxed-math -cl-no-signed-zeros
    return module.get_function('calcO')

