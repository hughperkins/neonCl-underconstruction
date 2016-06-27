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
#ctx = cl.create_some_context()
q = cl.CommandQueue(ctx)

mf = cl.mem_flags

k_calcU = winograd_kernels_cl.calcU(ctx)
k_calcV = winograd_kernels_cl.calcV(ctx)
k_calcM = winograd_kernels_cl.calcM(ctx)
# xprop_image_trans_4x4_kernel = winograd_kernels_cl.xprop_image_trans_4x4_kernel(ctx)

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

def calcU(q, W):
    Ci = W.shape[0]
    kH = W.shape[1]
    kW = W.shape[2]
    Co = W.shape[3]

    # this is adapted from neon's winograd_conv.py:
    GK   = ceil_div(Co, 32)

    filter_size   = 1152*Ci*GK
    grid = (GK, Ci, 1)
    block = (32, 1, 1)
    W_cl = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=W)
    U_from_cl = np.zeros((filter_size,), dtype=np.float32)
    print('size U_from_cl', U_from_cl.size)
    U_cl = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=U_from_cl)
    q.finish()
    timecheck('created U_cl buffers')
    
    call_cl_kernel(
        k_calcU,
        q, grid, block,
        U_cl, W_cl,
        kH * kW * Co, kW * Co, kW * Co * 2, Co, Ci * 1152,
        Ci, GK)
    print('Ci', Ci, 'Co', Co, 'W.shape', W.shape, 'W.size', W.size, 'U_from_cl.size', U_from_cl.size)
    q.finish()
    timecheck('calced U_cl')
    cl.enqueue_copy(q, U_from_cl, U_cl)
    #print('GK', GK, 'Ci', Ci, 'filter_size', filter_size, 'U_from_cl.size', U_from_cl.size)
    U_from_cl = U_from_cl.reshape(6,6,GK,Ci,32)#[:Co,:,:,0]
    #U_from_cl = np.transpose(U_from_cl, [2,3,0,4,1]).reshape(6, 6, GK * 32, Ci)[:,:,:Co,:]
    # assert np.allclose(U_from_cl, U2, atol=1e-4)

    # layout
    # [xi, nu, Co // 32, Ci, Co % 32]

    U_from_cpu = winograd_cpu.calcU(W=W)
    U_from_cl_ = U_from_cl.transpose(0,1,2,4,3).reshape(6,6,GK * 32,Ci)[:,:,:Co,:]

    assert np.allclose(U_from_cl_, U_from_cpu, atol=1e-4)

    return U_from_cl
    #return U_from_cpu

def calcV(I):
    Ifull = I
    Ci = I.shape[0]
    iH = I.shape[1]
    iW = I.shape[2]
    N = I.shape[3]
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

    div_GXS2 = get_div_mul_shift_64(GXS2)

    image_size = 1152*Ci*GXS*GYS*GN
    V_from_cl = np.zeros((image_size,), dtype=np.float32)
    I_cl = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=I)
    V_cl = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=V_from_cl)
    q.finish()
    timecheck('allocated V_cl buffers')

    grid = (GN, GYS*GXS, Ci)
    block = (32, 1, 1)

    call_cl_kernel(
        k_calcV,
        q, grid, block,
        V_cl, I_cl,
        
        iH, iW, N, padH, padW,
        GXS, GYS2, GXS2, div_GXS2[0], div_GXS2[1],
        shlY, shlX, maskY, shrY, maskX, shrX, shlN, maskN,
        iH * iW * N, iW * N, GYS*GXS*Ci*1152, GXS * Ci * 1152, Ci * 1152,
        GXS, GXS * GYS, GN, Ci)
    q.finish()
    timecheck('calced V_cl')

    cl.enqueue_copy(q, V_from_cl, V_cl)
    print('image_size', image_size)
    print('GXS', GXS, 'GYS', GYS, 'GN', GN, 'Ci', Ci)
    V_from_cl = V_from_cl.reshape(6,6,GYS,GXS,GN,Ci,32)
    # [xi, nu, n // 32, th, tw, ci, n % 32]

    V_from_cpu = winograd_cpu.calcV(I=I)
    # [n, xi, nu, ci, th, tw]]

    V_from_cl_ = V_from_cl.transpose(
        2,6,0,1,5,3,4).reshape(
        GN * 32, 6, 6, Ci, tiles, tiles)[:N]

    #V_from_cl = np.transpose(V_from_cl, [2, 6, 4, 5, 3, 0, 1])
    #V_from_cl = V_from_cl.reshape(GN * 32, 6, 6, Ci, tiles, tiles)[:N,:,:,:,:,:]

    assert np.allclose(V_from_cl_, V_from_cpu, atol=1e-4)

    return V_from_cl

def calcM(N, Co, U, V):
    # U_from_cl.reshape(GK,Ci,6,6,32)  6,6,GK,Ci,32
    print('U.shape', U.shape)
    Co = (U.shape[2] - 1) * 32 + U.shape[4]
    Ci = U.shape[3]
    GK   = ceil_div(Co, 32)
    tiles = V.shape[4]
    GN = V.shape[2]
    print('GK', GK, 'GN', GN, 'tiles', tiles, 'Co', Co, 'Ci', Ci)

    # M_from_cl = np.zeros((GK * 32 * GN * 32 * tiles * tiles * 6 * 6,), dtype=np.float32)
    M_from_cl = np.zeros((GK, 32, GN, 32, tiles, tiles, 6, 6,), dtype=np.float32)
    U_cl = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=U)
    V_cl = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=V)
    M_cl = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=M_from_cl)
    q.finish()
    timecheck('allocated M_cl buffers')

    # grid = (GN, GYS*GXS, Ci)
    grid = (1,1,1) # for now...
    block = (32, 1, 1)

    call_cl_kernel(
        k_calcM,
        q, grid, block,
        M_cl, U_cl, V_cl,
        
        Ci, 1)
    q.finish()
    timecheck('calced M_cl')

    cl.enqueue_copy(q, M_from_cl, M_cl)

    M_cpu = winograd_cpu.calcM(N=N, Co=Co, U=U, V=V)

    M_from_cl_ = M_from_cl.reshape(GK * 32, GN * 32, tiles, tiles, 6, 6)
    print('M_cpu.shape', M_cpu.shape)
    print('M_from_cl_.shape', M_from_cl_.shape)

    #M_from_cl_ = M_from_cl.transpose()
    print(M_from_cl_[0,0,0,0])
    print(M_cpu[0,0,0,0])
    print(M_from_cl_[0,1,0,0])
    print(M_cpu[0,1,0,0])
    print(M_from_cl_[1,0,0,0])
    print(M_cpu[1,0,0,0])
    sys.exit(1)

    # old layouts:
    # U                           Co // 32,       Ci,    6,   6, Co % 32
                         # bytes:           eg 150KB, 4.6K, 768,     128
    # V            # tiles, tiles, N // 32,       Ci,    6,   6,  N % 32
            # bytes                         eg 150KB, 4.6K, 768,     128

    # new layouts:
    # U
    # [xi, nu, co // 32,         ci, co % 32]
    # V
    # [xi, nu,  n // 32, th, tw, ci,  n % 32]

    M_cpu_blocked_l1 = winograd_cpu.calcM_blocked_l1(N=N, Co=Co, U=U, V=V)
    assert np.allclose(M_cpu, M_cpu_blocked_l1, atol=1e-3)

    return M_cpu

def process(iH, iW, N, Ci, Co, kH=3, kW=3):
    inittime()
    np.random.seed(123)

    oH = iH
    oW = iW

    W = np.random.randn(Ci,kH,kW,Co).astype(np.float32)

    I = np.zeros((Ci,iH, iW,N), dtype=np.float32)
    I[:] = np.random.randn(*I.shape)

    print('Co', Co, 'iH', iH, 'iW', iW, 'N', N)

    U = calcU(q=q, W=W)
    V = calcV(I=I)
    M = calcM(N=N, Co=Co, U=U, V=V)

    O = winograd_cpu.calcO(M=M)

    return {'W': W, 'O': O, 'I': I}

def simple1():
    image_size = 4
    N = 32
    Ci = 32
    Co = 32
 
    start = time.time()
    for it in range(5):
        res = process(iH=image_size, iW=image_size, N=N, Ci=Ci,
            Co=Co)
    end = time.time()
    print('diff', end - start)
    O = res['O']
    I = res['I']
    W = res['W']
    # print('wino O[0,:,:,0]')

    cpuO = np.zeros((Co, image_size, image_size, N), dtype=np.float32)
    for n in range(N):
        for co in range(Co):
            for h in range(image_size):
                for w in range(image_size):
                    cpuvalue = cpu_check.checkO(W=W, I=I, O=O, c=co, h=h, w=w, n=n)
                    cpuO[co, h, w, n] = cpuvalue
    #print('cpuO[0]', cpuO[0])
    n_values = np.random.choice(N, (min(N, 3),), False)
    co_values = np.random.choice(Co, (min(Co, 3),), False)
    for n in n_values:
      for co in co_values:
        #print('co', co, 'n', n)
        #print('winograd')
        #print(O[co,:,:,n].reshape(image_size, image_size))
        #print('cpu')
        #print(cpuO[co,:,:,n].reshape(image_size,image_size))
        assert np.allclose(O[co,:,:,n], cpuO[co,:,:,n], atol=1e-3)
    #printTensor(cpuO[0])

    print('diff', end - start)

   # checkO(W=W, I=I, O=O, c=0, h=0, w=0, n=0)
   # checkO(W=W, I=I, O=O, c=0, h=0, w=0, n=1)
   # checkO(W=W, I=I, O=O, c=0, h=1, w=0, n=0)
   # checkO(W=W, I=I, O=O, c=0, h=0, w=1, n=0)
   # checkO(W=W, I=I, O=O, c=1, h=0, w=0, n=0)
#    checkO(W=W, I=I, O=O, c=3, h=2, w=1, n=27)

np.set_printoptions(precision=2, suppress=True)
simple1()

