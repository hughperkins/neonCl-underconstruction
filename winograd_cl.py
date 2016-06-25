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

import numpy as np
import time
import pyopencl as cl
from neoncl import api
import pyopencl as cl
from neoncl.util.math_helper import get_div_mul_shift_32, get_div_mul_shift_64, ceil_div
import winograd_kernels_cl


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

fprop_filter_trans_kernel = winograd_kernels_cl.get_fprop_filter_trans_kernel(ctx)

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

def printDims(W, I):
    Ci = W.shape[0]
    iH = I.shape[1]
    iW = I.shape[2]
    Co = W.shape[3]
    kH = W.shape[1]
    kW = W.shape[2]
    print('Ci', Ci, 'iH', iH, 'iW', iW, 'Co', Co, 'kH', kH, 'kW', kW)

def check_gradWeights(O, I, W, gradO, gradW, ci, h, w, co, eps=1e-2):
#    eps = 1e4 #hack
    N = I.shape[3]
    iH = I.shape[1]
    iW = I.shape[2]
    Ci = W.shape[0]
    kH = W.shape[1]
    kW = W.shape[2]
    Co = W.shape[3]
    oH = iH # assuming padded, which it is
    oW = iW # assuming padded, which it is
#    print('Ci', Ci, 'iH', iH, 'iW', iW, 'Co', Co, 'kH', kH, 'kW', kW)

#    ih = h
#    iw = w
    kh = h
    kw = w
#    ci = c

    padw = 1
    padh = 1

    sum = 0

    for ow in range(oW):
        for oh in range(oH):
            ih = oh + kh - padh
            iw = ow + kw - padw
            for n in range(N):
                if ih >= 0 and iw >= 0 and ih < iH and iw < iW:
                    v = I[ci, ih, iw, n] * gradO[co * iH * iW + oh * iW + ow, n]
                    sum += v
    cpu_value = sum
    gpu_value = gradW[ci, kh, kw, co]
    diff =  abs(cpu_value - gpu_value)
    print('checkGradW gpu=%.6f cpu=%.6f diff=%.6f' % (gpu_value, cpu_value, diff))
    assert abs(cpu_value - gpu_value) < eps

def check_gradI(O, I, W, gradO, gradI, c, h, w, n, eps=1e-4):
    N = I.shape[3]
    iH = I.shape[1]
    iW = I.shape[2]
    Ci = W.shape[0]
    kH = W.shape[1]
    kW = W.shape[2]
    Co = W.shape[3]
    oH = iH # assuming padded, which it is
    oW = iW # assuming padded, which it is
#    print('Ci', Ci, 'iH', iH, 'iW', iW, 'Co', Co, 'kH', kH, 'kW', kW)

    ih = h
    iw = w
    ci = c

    padw = 1
    padh = 1

    sum = 0
    for co in range(Co):
        for kh in range(kH):
            for kw in range(kW):
                ow = iw - kw + padw
                oh = ih - kh + padh
                if ow >= 0 and oh >= 0 and ow < oW and oh < oH:
                    v = gradO[co * iH * iW + oh * iW + ow, n] * W[ci, kh, kw, co]
                    sum += v
    cpu_value = sum
    gpu_value = gradI[c, ih, iw, n]
    diff =  abs(cpu_value - gpu_value)
    print('checkGradI gpu=%.6f cpu=%.6f diff=%.6f' % (gpu_value, cpu_value, diff))
    assert abs(cpu_value - gpu_value) < eps

def checkO(O, W, I, c, h, w, n, eps=1e-4):
    Ci = W.shape[0]
    iH = I.shape[1]
    iW = I.shape[2]
    Co = W.shape[3]
    kH = W.shape[1]
    kW = W.shape[2]
#    print('Ci', Ci, 'iH', iH, 'iW', iW, 'Co', Co, 'kH', kH, 'kW', kW)

    co = c
    padw = 1
    padh = 1

    # we are going to apply entire kernel, over all input channels, to the input
    # image, in one location
    sum = 0
    for kw in range(kW):
        for kh in range(kH):
            for ci in range(Ci):
                ih = h + kh - padh
                iw = w + kw - padw
                if ih >= 0 and iw >= 0 and ih < iH and iw < iW:
                    v = I[ci, ih, iw, n] * W[ci, kh, kw, co]
                    sum += v
    cpu_value = sum
    gpu_value = O[c, h, w,n]
    diff =  abs(cpu_value - gpu_value)
    # print('checkO c', c, 'h', h, 'w', w, 'n', n, 'cpu %.6f gpu %.6f diff %.6f' % (sum, gpu_value, diff))
    # assert diff < eps
    return cpu_value

last = 0
def inittime():
    global last
    last = time.time()

def timecheck(label):
    global last
    now = time.time()
    print(label, '%.2f' % ((now - last) * 1000))
    last = now

def process_one(iH, iW, Ci, Co, n, kH, kW, I, U, O):
    oH = iH
    oW = iW

    tiles = iW // 4
    BT = np.array([[4,0,-5,0,1,0],
          [0,-4,-4,1,1,0],
          [0,4,-4,-1,1,0],
          [0,-2,-1,2,1,0],
          [0,2,-1,-2,1,0],
          [0,4,0,-5,0,1]], dtype=np.float32)

    AT = np.array([[1,1,1,1,1,0],
        [0,1,-1,2,-2,0],
        [0,1,1,4,4,0],
        [0,1,-1,8,-8,1]], dtype=np.float32)

    Ifull = I
    # Wfull = W
    Ofull = O
    timecheck('allocated BT G AT')

    V2 = np.zeros((6, 6, Ci, tiles, tiles), dtype=np.float32)
    timecheck('allocaed V2')
    V = np.zeros((6, 6), dtype=np.float32) # transformed image
    Vtmp = np.zeros((6,6), dtype=np.float32)
    for th in range(tiles):
        hstart = -1 + 4 * th
        hend = hstart + 6 - 1
        hstarttrunc = max(0, hstart)
        hendtrunc = min(hend, iH - 1)
        hstartoffset = hstarttrunc - hstart
        hendoffset = hendtrunc - hstart
        for tw in range(tiles):
            wstart = -1 + 4 * tw
            wend = wstart + 6 - 1
            wstarttrunc = max(0, wstart)
            wendtrunc = min(wend, iW - 1)
            wstartoffset = wstarttrunc - wstart
            wendoffset = wendtrunc - wstart
            Ipadded = np.zeros((6, 6), dtype=np.float32)
            for ci in range(Ci):
                Ipadded[hstartoffset:hendoffset + 1,wstartoffset:wendoffset + 1] = Ifull[ci,hstarttrunc:hendtrunc+1,wstarttrunc:wendtrunc+1,n]
                I = Ipadded
                #for i in range(6):
                    #Vtmp[0][i] = + 4 * I[0][i] - 5 * I[2][i]               + I[4][i]
                    #Vtmp[1][i] = - 4 * I[1][i] - 4 * I[2][i] +     I[3][i] + I[4][i]
                    #Vtmp[2][i] = + 4 * I[1][i] - 4 * I[2][i] -     I[3][i] + I[4][i]
                    #Vtmp[3][i] = - 2 * I[1][i] -     I[2][i] + 2 * I[3][i] + I[4][i]
                    #Vtmp[4][i] = + 2 * I[1][i] -     I[2][i] - 2 * I[3][i] + I[4][i]
                    #Vtmp[5][i] = + 4 * I[1][i]               - 5 * I[3][i]           + I[5][i]
                Vtmp = BT.dot(I)

                # each i is a row of V
                #for i in range(6):
                    #V[i][0] = + 4 * Vtmp[i][0] - 5 * Vtmp[i][2]           + Vtmp[i][4]
                    #V[i][1] = - 4 * Vtmp[i][1] - 4 * Vtmp[i][2] +     Vtmp[i][3] + Vtmp[i][4]
                    #V[i][2] = + 4 * Vtmp[i][1] - 4 * Vtmp[i][2] -     Vtmp[i][3] + Vtmp[i][4]
                    #V[i][3] = - 2 * Vtmp[i][1] -     Vtmp[i][2] + 2 * Vtmp[i][3] + Vtmp[i][4]
                    #V[i][4] = + 2 * Vtmp[i][1] -     Vtmp[i][2] - 2 * Vtmp[i][3] + Vtmp[i][4]
                    #V[i][5] = + 4 * Vtmp[i][1]               - 5 * Vtmp[i][3]           + Vtmp[i][5]
                V2[:,:,ci,th,tw] = Vtmp.dot(BT.T)
                #V = Vtmp.dot(BT.T)
                #V2[:,:,ci,th,tw] = V
#                for i in range(6):
 #                   for j in range(6):
  #                      V2[i, j, ci, th, tw] = V[i, j]
    timecheck('calced V2')

    M = np.zeros((Co, tiles, tiles, 6, 6), dtype=np.float32)
    for mh in range(6):
        for mw in range(6):
            #print('U2[mh,mw].shape', U2[mh,mw].shape, V2[mh,mw].shape)
            M[:, :, :, mh, mw] = np.tensordot(U[mh,mw], V2[mh,mw], 1)
            # res = np.tensordot(U2[mh,mw], V2[mh,mw], 1)
            #print('res.shape', res.shape)
            # M[:, :, :, mh, mw] = res
    timecheck('calced M')

    Mfull = M
    # inverse transform
    Otmp = np.zeros((4, 6), dtype=np.float32)
    for co in range(Co):
        for th in range(tiles):
            for tw in range(tiles):
                O = Ofull[co,th * 4:(th+1)*4,tw*4:(tw+1)*4,n].reshape(4,4)
                M = Mfull[co, th, tw]
                #for i in range(6):
                    #Otmp[0][i] = M[0][i] + M[1][i] + M[2][i] + M[3][i] + M[4][i]
                    #Otmp[1][i] =         + M[1][i] - M[2][i] + 2 * M[3][i] - 2 * M[4][i]
                    #Otmp[2][i] =         + M[1][i] + M[2][i] + 4 * M[3][i] + 4 * M[4][i]
                    #Otmp[3][i] =         + M[1][i] - M[2][i] + 8 * M[3][i] - 8 * M[4][i] + M[5][i]
                    #print('AT.shape', AT.shape, 'M.shape', M.shape)
                Otmp = AT.dot(M)

                #for i in range(4):
                    #O[i][0] = Otmp[i][0] + Otmp[i][1] + Otmp[i][2] + Otmp[i][3] + Otmp[i][4]
                    #O[i][1] =         + Otmp[i][1] - Otmp[i][2] + 2 * Otmp[i][3] - 2 * Otmp[i][4]
                    #O[i][2] =         + Otmp[i][1] + Otmp[i][2] + 4 * Otmp[i][3] + 4 * Otmp[i][4]
                    #O[i][3] =         + Otmp[i][1] - Otmp[i][2] + 8 * Otmp[i][3] - 8 * Otmp[i][4] + Otmp[i][5]
                    #print('O.shape', O.shape, 'Otmp.shape', Otmp.shape, 'AT.T.shape', AT.T.shape)
                O[:] = Otmp.dot(AT.T)
    timecheck('calced O')

def calcU(q, W):
    G = np.array([[1/4,0,0],
        [-1/6,-1/6,-1/6],
        [-1/6,1/6,-1/6],
        [1/24,1/12,1/6],
        [1/24,-1/12,1/6],
        [0,0,1]], dtype=np.float32)

    Ci = W.shape[0]
    Co = W.shape[3]

    Wfull = W

    U2 = np.zeros((6, 6, Co, Ci), dtype=np.float32)
    Utmp = np.zeros((6, 3), dtype=np.float32)
    U = np.zeros((6, 6), dtype=np.float32)  # transformed filter
    timecheck('allocaed U')

    for co in range(Co):
        for ci in range(Ci):
            W = Wfull[ci,:,:,co].reshape(3,3)
            #for i in range(3):
                #Utmp[0][i] = 1/4 * W[0][i]
                #Utmp[1][i] = - 1/6 * (W[0][i] + W[1][i] + W[2][i])
                #Utmp[2][i] = - 1/6 *W[0][i] + 1/6 * W[1][i] - 1/6 * W[2][i]
                #Utmp[3][i] = 1/24 * W[0][i] + 1/12 * W[1][i] + 1/6 * W[2][i]
                #Utmp[4][i] = 1/24 * W[0][i] - 1/12 * W[1][i] + 1/6 * W[2][i]
                #Utmp[5][i] = W[2][i]
            Utmp = G.dot(W)

            #for i in range(6):
                #U[i][0] = 1/4 * Utmp[i][0]
                #U[i][1] = - 1/6 * Utmp[i][0] - 1/6 * Utmp[i][1] - 1/6 * Utmp[i][2]
                #U[i][2] = - 1/6 * Utmp[i][0] + 1/ 6 * Utmp[i][1] - 1 / 6 * Utmp[i][2]
                #U[i][3] = 1/24 * Utmp[i][0] + 1/12 * Utmp[i][1] + 1/6 * Utmp[i][2]
                #U[i][4] = 1/24 * Utmp[i][0] - 1/12 * Utmp[i][1] + 1/6 * Utmp[i][2]
                #U[i][5] = Utmp[i][2]
            U = Utmp.dot(G.T)

            U2[:,:,co,ci] = U
            #for i in range(6):
            #    for j in range(6):
            #        U2[i, j, co, ci] = U[i, j]
    timecheck('calced U2')
    # print('U from python', U2)
    return U2

    # this is adapted from neon's winograd_conv.py:
    if N == 1:
        shiftN = 0
    elif N < 32:
        shiftN = len(bin(N-1))-2
    else:
        shiftN = 5
    blkN = 1 << shiftN

    shiftY, shiftX, superY, superX = {
        1 : (3,4,0x203,0x300), # 4x8
        2 : (3,3,0x203,0x201), # 4x4
        4 : (2,3,0x104,0x202), # 2x4
        8 : (2,2,0x104,0x103), # 2x2
        16: (1,2,0x000,0x104), # 1x2
        32: (1,1,0x000,0x000), # 1x1
    }.get(blkN)

    gridCo = ceil_div(Co, 32)
    # gridY = ceil_div(oH, 1<<shiftY)
    # gridX = ceil_div(oY, 1<<shiftX)
    # gridN = ceil_div(N, blkN)
    # Y2    = gridY // 2
    # X2    = gridX  * 2

    dtype_itemsize = 4
    trans_size   = C * gridCo * 512 * dtype_itemsize
    trans_shared = 512 * dtype_itemsize
    trans_args   = [(gridK, C, 1), (32, 1, 1), None,
                     None, None, kH*kW*Co, kW*Co, kW*Co*2, Co]
    W_cl = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=W)
    U_from_cl = np.zeros((6, 6, Co, Ci), dtype=np.float32)
    U_cl = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=U_from_cl)
    
    fprop_filter_trans_kernel(
        q, U_cl, W_cl, kH * kW * Co, kW * Co, kW * Co * 2,
        Co,
        cl.LocalMemory(trans_shared))

    cl.enqueue_copy(q, U_from_cl, U_cl)
    print('U_from_cl', U_from_cl)
    return U2

def process(iH, iW, N, Ci, Co, kH=3, kW=3):
    inittime()
    np.random.seed(123)

    oH = iH
    oW = iW

    W = np.random.randn(Ci,kH,kW,Co).astype(np.float32)

    Wfull = W

    I = np.zeros((Ci,iH, iW,N), dtype=np.float32)
    I[:] = np.random.randn(*I.shape)
    Ifull = I

    print('Co', Co, 'iH', iH, 'iW', iW, 'N', N)
    O = np.zeros((Co, oH, oW, N), dtype=np.float32)
    Ofull = O

    U = calcU(q=q, W=W)
    for n in range(N):
        print('n', n)
        process_one(iH=iH, iW=iW, Ci=Ci, Co=Co, kH=kH, kW=kW, n=n, I=I, U=U, O=O)

    I = Ifull
    W = Wfull
    O = Ofull
    return {'W': W, 'O': O, 'I': I}

def simple1():
    image_size = 16
    N = 4
    Ci = 16
    Co = 16
 
    start = time.time()
    for it in range(5):
        res = process(iH=image_size, iW=image_size, N=N, Ci=Ci,
            Co=Co)
    end = time.time()
    print('diff', end - start)
    np.set_printoptions(precision=2, suppress=True)
    O = res['O']
    I = res['I']
    W = res['W']
    # print('wino O[0,:,:,0]')

    cpuO = np.zeros((Co, image_size, image_size, N), dtype=np.float32)
    for n in range(N):
        for co in range(Co):
            for h in range(image_size):
                for w in range(image_size):
                    cpuvalue = checkO(W=W, I=I, O=O, c=co, h=h, w=w, n=n)
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
        assert np.allclose(O[co,:,:,n], cpuO[co,:,:,n], atol=1e-4)
    #printTensor(cpuO[0])

    print('diff', end - start)

   # checkO(W=W, I=I, O=O, c=0, h=0, w=0, n=0)
   # checkO(W=W, I=I, O=O, c=0, h=0, w=0, n=1)
   # checkO(W=W, I=I, O=O, c=0, h=1, w=0, n=0)
   # checkO(W=W, I=I, O=O, c=0, h=0, w=1, n=0)
   # checkO(W=W, I=I, O=O, c=1, h=0, w=0, n=0)
#    checkO(W=W, I=I, O=O, c=3, h=2, w=1, n=27)

simple1()

