# Copyright 2016 Hugh Perkins, 2014 Nervana Systems Inc. All rights reserved.
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

from __future__ import print_function, division
import time
import random
import numpy as np
import pyopencl as cl
from neoncl import api
import pyopencl as cl
from neoncl.util.math_helper import get_div_mul_shift_32, get_div_mul_shift_64, ceil_div
import winograd_kernels_cl
import winograd_cpu
from neoncl.backends.kernels.cl.callkernel import call_cl_kernel
import cpu_check
from timecheck import inittime, timecheck


gpu_idx = 0

platforms = cl.get_platforms()
i = 0
for platform in platforms:
   gpu_devices = platform.get_devices(device_type=cl.device_type.GPU)
   if gpu_idx < i + len(gpu_devices):
       ctx = cl.Context(devices=[gpu_devices[gpu_idx-i]])
       break
   i += len(gpu_devices)

print('context', ctx)
q = cl.CommandQueue(ctx)

mf = cl.mem_flags

k_calcU = winograd_kernels_cl.calcU(ctx)
k_calcV = winograd_kernels_cl.calcV(ctx)
k_calcM = winograd_kernels_cl.calcM(ctx)
k_calcO = winograd_kernels_cl.calcO(ctx)

its = 1

mf = cl.mem_flags

def printTensor(t):
   dims = len(t.shape)
   print('dims', dims)
   if dims == 3:
      for i in range(t.shape[0]):
         print('[%s, ...' % i)
         for x in range(t.shape[1]):
            line = ''
            for y in range(t.shape[2]):
               line += '%.1f ' % t[i][x][y]
            print(line)

def calcU(q, W_shape, W_cl, U_cl):
    Ci = W_shape[0]
    kH = W_shape[1]
    kW = W_shape[2]
    Co = W_shape[3]

    # this is adapted from neon's winograd_conv.py:
    GK   = ceil_div(Co, 32)

    filter_size   = 1152*Ci*GK
    grid = (GK, Ci, 1)
    block = (32, 1, 1)
    
    call_cl_kernel(
        k_calcU,
        q, grid, block,
        U_cl, W_cl,
        kH * kW * Co, kW * Co, kW * Co * 2, Co, Ci * 1152,
        Ci, GK)
    q.finish()
    timecheck('calced U_cl')

def calcV(I_shape, I_cl, V_cl):
    #Ifull = I
    Ci = I_shape[0]
    iH = I_shape[1]
    iW = I_shape[2]
    N = I_shape[3]
    tiles = iW // 4

    oH = iH
    oW = iW
    padH = 1
    padW = 1

    # adapted from winograd_conv.py
    #if N == 1:
    #    shlN = 0
    #elif N < 32:
    #    shlN = len(bin(N-1))-2
    #else:
    #    shlN = 5
    shlN = 5
    shlY, shlX, maskY, shrY, maskX, shrX, maskN, supY, supX = {
        0 : (4, 5, 0x18, 3, 0x07, 0, 0x00, 0x203, 0x300), # 4x8  yyxxx
        1 : (4, 4, 0x18, 3, 0x06, 1, 0x01, 0x203, 0x201), # 4x4  yyxxn
        2 : (3, 4, 0x10, 4, 0x0c, 2, 0x03, 0x104, 0x202), # 2x4  yxxnn
        3 : (2, 4, 0x00, 0, 0x18, 3, 0x07, 0x000, 0x203), # 1x4  xxnnn
        4 : (2, 3, 0x00, 0, 0x10, 4, 0x0f, 0x000, 0x104), # 1x2  xnnnn
        5 : (2, 2, 0x00, 0, 0x00, 0, 0x1f, 0x000, 0x000), # 1x1  nnnnn
    }.get(shlN)

    GYS  = ceil_div(oH, 1 << shlY)
    GXS  = ceil_div(oW, 1 << shlX)
    GN   = ceil_div(N, 1 << shlN)
    # GK   = ceil_div(Co, 32)
    GYS2 = GYS // 2
    GXS2 = GXS  * 2

    div_GXS2 = get_div_mul_shift_32(GXS * GYS, GXS2)
    div_GXS = get_div_mul_shift_32(GXS * GYS, GXS)

    image_size = 1152*Ci*GXS*GYS*GN
    
    print('div_GXS', div_GXS)

    print('GYS', GYS, 'GXS', GXS, 'GN', GN, 'Ci', Ci, 'GY_GX', GXS * GYS)
    grid = (GN, GYS*GXS, Ci)
    block = (32, 1, 1)

    call_cl_kernel(
        k_calcV,
        q, grid, block,
        V_cl, I_cl,
        
        iH, iW, N, padH, padW,
        GXS, GYS2, GXS2, div_GXS2[0], div_GXS2[1], div_GXS[0], div_GXS[1],
        shlY, shlX, maskY, shrY, maskX, shrX, shlN, maskN,
        iH * iW * N, iW * N, GYS*GXS*Ci*1152, GXS * Ci * 1152, Ci * 1152,
        GXS, GXS * GYS, GN, Ci)
    q.finish()
    timecheck('calced V_cl')

def calcM(N, Co, M_cl, U_shape, U_cl, V_shape, V_cl):
    Co = (U_shape[2] - 1) * 32 + U_shape[4]
    Ci = U_shape[3]
    GK   = ceil_div(Co, 32)
    tiles = V_shape[4]
    GN = V_shape[2]
    print('GK', GK, 'GN', GN, 'tiles', tiles, 'Co', Co, 'Ci', Ci, 'N', N)

    grid = (tiles * tiles,1,1) # b
    block = (32, 1, 1)

    call_cl_kernel(
        k_calcM,
        q, grid, block,
        M_cl, U_cl, V_cl,
        
        Ci, 1, tiles, GN, GK,
        cl.LocalMemory(32 * 32 * 4), cl.LocalMemory(32 * 32 * 4))
    q.finish()
    timecheck('calced M_cl')

    # new layouts:
    # U
    # [xi, nu, co // 32,         ci, co % 32]
    # V
    # [xi, nu,  n // 32, th, tw, ci,  n % 32]

    # [n//32][n % 32][co // 32][co % 32][th][tw][xi][nu] 

def calcO(O_cl, M_shape, M_cl):
    GK = M_shape[2]
    GN = M_shape[0]
    tiles = M_shape[4]

    num_xinu_tiles = GK * 32 * GN * 32 * tiles * tiles
    grid = (ceil_div(num_xinu_tiles, 32), 1, 1)
    block = (32, 1, 1)

    call_cl_kernel(
        k_calcO,
        q, grid, block,
        O_cl, M_cl,
        num_xinu_tiles
    )
    q.finish()
    timecheck('calced O_cl')

def process(iH, iW, N, Ci, Co, kH=3, kW=3):
    inittime()
    np.random.seed(123)

    oH = iH
    oW = iW
    
    tiles = iW // 4

    shlN = 5
    shlY, shlX, maskY, shrY, maskX, shrX, maskN, supY, supX = {
        0 : (4, 5, 0x18, 3, 0x07, 0, 0x00, 0x203, 0x300), # 4x8  yyxxx
        1 : (4, 4, 0x18, 3, 0x06, 1, 0x01, 0x203, 0x201), # 4x4  yyxxn
        2 : (3, 4, 0x10, 4, 0x0c, 2, 0x03, 0x104, 0x202), # 2x4  yxxnn
        3 : (2, 4, 0x00, 0, 0x18, 3, 0x07, 0x000, 0x203), # 1x4  xxnnn
        4 : (2, 3, 0x00, 0, 0x10, 4, 0x0f, 0x000, 0x104), # 1x2  xnnnn
        5 : (2, 2, 0x00, 0, 0x00, 0, 0x1f, 0x000, 0x000), # 1x1  nnnnn
    }.get(shlN)

    GYS  = ceil_div(oH, 1 << shlY)
    GXS  = ceil_div(oW, 1 << shlX)
    GN   = ceil_div(N, 1 << shlN)
    # GK   = ceil_div(Co, 32)
    GYS2 = GYS // 2
    GXS2 = GXS  * 2

    GK   = ceil_div(Co, 32)

    W = np.random.randn(Ci,kH,kW,Co).astype(np.float32)

    I = np.zeros((Ci,iH, iW,N), dtype=np.float32)
    I[:] = np.random.randn(*I.shape)

    print('Co', Co, 'iH', iH, 'iW', iW, 'N', N, 'tiles', tiles)

    W_cl = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=W)
    I_cl = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=I)

    U = np.zeros((6, 6, GK, Ci, 32,), dtype=np.float32)
    U_cl = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=U)

    V = np.zeros((6, 6, GN,GXS, GYS, Ci, 32), dtype=np.float32)
    V_cl = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=V)

    M = np.zeros((GN, 32, GK, 32, tiles, tiles, 6, 6,), dtype=np.float32)
    M_cl = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=M)

    O = np.zeros((GN, 32, GK, 32, tiles, tiles, 4, 4,), dtype=np.float32)
    O_cl = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=O)

    q.finish()
    print('allocated buffers')
    start = time.time()

    for it in range(3):
        calcU(q=q, U_cl=U_cl, W_shape=W.shape, W_cl=W_cl)
        calcV(V_cl=V_cl, I_shape=I.shape, I_cl=I_cl)
        calcM(N=N, Co=Co, M_cl=M_cl, U_shape=U.shape, U_cl=U_cl, V_shape=V.shape, V_cl=V_cl)

        calcO(O_cl=O_cl, M_shape=M.shape, M_cl=M_cl)

        q.finish()
        end = time.time()
        print('calcs done')
        print('time for all calcs:', end - start)
        start = time.time()

    cl.enqueue_copy(q, O, O_cl)

    O = O.transpose(2,3, 4,6, 5,7, 0,1).reshape(
        GK * 32, tiles * 4, tiles * 4, GN * 32)
    print('O.shape', O.shape)

    W_from_cl = np.zeros((Ci, 3, 3, Co), dtype=np.float32)
    cl.enqueue_copy(q, W_from_cl, W_cl)

    U_from_cpu = winograd_cpu.calcU(W=W)
    U_from_cl = np.zeros((6, 6, GK, Ci, 32), dtype=np.float32)
    cl.enqueue_copy(q, U_from_cl, U_cl)
    U_from_cl_ = U_from_cl.transpose(
        0, 1, 2, 4, 3).reshape(6, 6, GK * 32, Ci)[:, :, :Co]
    assert np.allclose(U_from_cl_, U_from_cpu, atol=1e-4)

    V_from_cpu = winograd_cpu.calcV(I=I)
    V_from_cl = np.copy(V)
    cl.enqueue_copy(q, V_from_cl, V_cl)
    
    print('tiles', tiles)
    # 0  1   2   3    4   5   6
    # 6, 6, GN,GXS, GYS, Ci, 32
    V_from_cl_ = V_from_cl.transpose(
        2,6,0,1,5,3,4).reshape(
        GN * 32, 6, 6, Ci, tiles, tiles)[:N]
    
    assert np.allclose(V_from_cl_, V_from_cpu, atol=1e-3)

    #      0       1         2        3   4   5   6   7
    # [n//32][n % 32][co // 32][co % 32][th][tw][xi][nu]
    M_from_cpu = winograd_cpu.calcM(U=U_from_cl, V=V_from_cl, N=N, Co=Co)
    M_from_cl = np.copy(M)
    cl.enqueue_copy(q, M_from_cl, M_cl)
    q.finish()
    M_from_cl = M_from_cl.reshape(GN * 32, GK * 32, tiles, tiles, 6, 6)[:N, :Co]
    
    print(M_from_cl.reshape(M_from_cl.size)[:20])
    
    assert np.allclose(M_from_cl, M_from_cpu, atol=1e-2)
    
    #np.transpose(V_from_cl, [2, 6, 4, 5, 3, 0, 1])
    #V_from_cl = V_from_cl.reshape(GN * 32, 6, 6, Ci, tiles, tiles)[:N,:,:,:,:,:]
    
    return {'W': W, 'O': O, 'I': I}

def simple1():
    #image_size = 16
    #N = 4
    #Ci = 256
    #Co = 4
 
    # {'padW': 1, 'kH': 3, 'iH': 56, 'iW': 56, 'padH': 1, 'kW': 3, 'Ci': 256, 'Co': 256, 'dW': 1, 'dH': 1}
    image_size = 56
    N = 32
    Ci = 32
    Co = 32

    #start = time.time()
    #for it in range(5):
    res = process(iH=image_size, iW=image_size, N=N, Ci=Ci,
        Co=Co)
    #end = time.time()
    #print('diff', end - start)
    O = res['O']
    I = res['I']
    W = res['W']

    random.seed(123)
    for it in range(400):
        n = random.randint(0,N - 1)
        co = random.randint(0, Co - 1)
        h = random.randint(0, image_size - 1)
        w = random.randint(0, image_size - 1)
        cpuvalue = cpu_check.checkO(W=W, I=I, O=O, c=co, h=h, w=w, n=n)
        diff = abs(O[co, h, w, n] - cpuvalue)
        if diff >= 1e-3:
            print('co', co, 'h', h, 'w', w, 'n', n, O[co, h, w, n], cpuvalue, 'diff', diff)
            assert False
        assert diff < 1e-3

np.set_printoptions(precision=2, suppress=True)
simple1()

