# Copyright 2016 Hugh Perkins, All rights reserved.
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

its = 1

mf = cl.mem_flags

# https://gist.github.com/lbn/836313e283f5d47d2e4e
#def matprint(mat, fmt="g"):
#    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
#    for x in mat:
#        for i, y in enumerate(x):
#            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
#        print("")

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

def process_one(iH, iW, Ci, Co, n, kH, kW, I, W, O):
    oH = iH
    oW = iW

    tiles = iW // 4
    print('tiles', tiles)

    BT = np.array([[4,0,-5,0,1,0],
          [0,-4,-4,1,1,0],
          [0,4,-4,-1,1,0],
          [0,-2,-1,2,1,0],
          [0,2,-1,-2,1,0],
          [0,4,0,-5,0,1]], dtype=np.float32)

    G = np.array([[1/4,0,0],
        [-1/6,-1/6,-1/6],
        [-1/6,1/6,-1/6],
        [1/24,1/12,1/6],
        [1/24,-1/12,1/6],
        [0,0,1]], dtype=np.float32)

    AT = np.array([[1,1,1,1,1,0],
        [0,1,-1,2,-2,0],
        [0,1,1,4,4,0],
        [0,1,-1,8,-8,1]], dtype=np.float32)
        
    # I = I.reshape(iH, iW)

    Ifull = I
    Wfull = W
    Ofull = O

    U2 = np.zeros((6, 6, Co, Ci), dtype=np.float32)
    for co in range(Co):
        for ci in range(Ci):
            print('FILTER ci', ci)
            Wall = W
            W = W[ci,:,:,co].reshape(3,3)
            #print('W.shape', W.shape)
            Utmp = np.zeros((6, 3), dtype=np.float32)
            for i in range(3):
                Utmp[0][i] = 1/4 * W[0][i]
                Utmp[1][i] = - 1/6 * (W[0][i] + W[1][i] + W[2][i])
                Utmp[2][i] = - 1/6 *W[0][i] + 1/6 * W[1][i] - 1/6 * W[2][i]
                Utmp[3][i] = 1/24 * W[0][i] + 1/12 * W[1][i] + 1/6 * W[2][i]
                Utmp[4][i] = 1/24 * W[0][i] - 1/12 * W[1][i] + 1/6 * W[2][i]
                Utmp[5][i] = W[2][i]
            print('Utmp', Utmp)
            print('GT.dot(W)', G.dot(W))

            U = np.zeros((6, 6), dtype=np.float32)  # transformed filter
            for i in range(6):
                U[i][0] = 1/4 * Utmp[i][0]
                U[i][1] = - 1/6 * Utmp[i][0] - 1/6 * Utmp[i][1] - 1/6 * Utmp[i][2]
                U[i][2] = - 1/6 * Utmp[i][0] + 1/ 6 * Utmp[i][1] - 1 / 6 * Utmp[i][2]
                U[i][3] = 1/24 * Utmp[i][0] + 1/12 * Utmp[i][1] + 1/6 * Utmp[i][2]
                U[i][4] = 1/24 * Utmp[i][0] - 1/12 * Utmp[i][1] + 1/6 * Utmp[i][2]
                U[i][5] = Utmp[i][2]
            print('U', U)
            print('(G.dot(W)).dot(G.T)', (G.dot(W)).dot(G.T))
            #W = W.reshape(Ci, kH, kW, Co)
            W = Wall

            for i in range(6):
                for j in range(6):
                    U2[i, j, co, ci] = U[i, j]

    V2 = np.zeros((6, 6, Ci, tiles, tiles), dtype=np.float32)
    for th in range(tiles):
        for tw in range(tiles):
            for ci in range(Ci):
                print('IMAGE ci', ci)
                # transform image
                Ipadded = np.zeros((6, 6), dtype=np.float32)
                wstart = -1 + 4 * tw
                wend = wstart + 6 - 1
                hstart = -1 + 4 * th
                hend = hstart + 6 - 1
                wstarttrunc = max(0, wstart)
                hstarttrunc = max(0, hstart)
                wendtrunc = min(wend, iW - 1)
                hendtrunc = min(hend, iH - 1)
                wstartoffset = wstarttrunc - wstart
                hstartoffset = hstarttrunc - hstart
                wendoffset = wendtrunc - wstart
                hendoffset = hendtrunc - hstart
                print('wstart', wstart, 'wend', wend, 'hstart', hstart, 'hend', hend)
                print('wstarttrunc', wstarttrunc, 'wendtrunc', wendtrunc, 'hstarttrunc', hstarttrunc, 'hendtrunc', hendtrunc)
                print('wstartoffset', wstartoffset, 'wendoffset', wendoffset, 'hstartoffset', hstartoffset, 'hendoffset', hendoffset)
                Ipadded[hstartoffset:hendoffset + 1,wstartoffset:wendoffset + 1] = Ifull[ci,hstarttrunc:hendtrunc+1,wstarttrunc:wendtrunc+1,n]
                print('Ifull', Ifull)
                print('Ipadded', Ipadded)
                I = Ipadded
                Vtmp = np.zeros((6,6), dtype=np.float32)
                for i in range(6):
                    Vtmp[0][i] = + 4 * I[0][i] - 5 * I[2][i]               + I[4][i]
                    Vtmp[1][i] = - 4 * I[1][i] - 4 * I[2][i] +     I[3][i] + I[4][i]
                    Vtmp[2][i] = + 4 * I[1][i] - 4 * I[2][i] -     I[3][i] + I[4][i]
                    Vtmp[3][i] = - 2 * I[1][i] -     I[2][i] + 2 * I[3][i] + I[4][i]
                    Vtmp[4][i] = + 2 * I[1][i] -     I[2][i] - 2 * I[3][i] + I[4][i]
                    Vtmp[5][i] = + 4 * I[1][i]               - 5 * I[3][i]           + I[5][i]
                print('Vtmp', Vtmp)
                print('BT.dot(I)', BT.dot(I))

                V = np.zeros((6, 6), dtype=np.float32) # transformed image
                # each i is a row of V
                for i in range(6):
                    V[i][0] = + 4 * Vtmp[i][0] - 5 * Vtmp[i][2]           + Vtmp[i][4]
                    V[i][1] = - 4 * Vtmp[i][1] - 4 * Vtmp[i][2] +     Vtmp[i][3] + Vtmp[i][4]
                    V[i][2] = + 4 * Vtmp[i][1] - 4 * Vtmp[i][2] -     Vtmp[i][3] + Vtmp[i][4]
                    V[i][3] = - 2 * Vtmp[i][1] -     Vtmp[i][2] + 2 * Vtmp[i][3] + Vtmp[i][4]
                    V[i][4] = + 2 * Vtmp[i][1] -     Vtmp[i][2] - 2 * Vtmp[i][3] + Vtmp[i][4]
                    V[i][5] = + 4 * Vtmp[i][1]               - 5 * Vtmp[i][3]           + Vtmp[i][5]
                print('V', V)
                print('(BT.dot(I)).dot(BT.t())', (BT.dot(I)).dot(BT.T))
                
                for i in range(6):
                    for j in range(6):
                        V2[i, j, ci, th, tw] = V[i, j]

                    # I = I.reshape(Ci, iH, iW, N)

                    # filters

    # M = np.zeros((N, Co, 1, 1)
    M = np.zeros((Co, tiles, tiles, oH + 2, oW + 2), dtype=np.float32)
    for mh in range(6):
        for mw in range(6):
            print('mh', mh, 'mw', mw)
            print('U2[mh,mw].shape', U2[mh,mw].shape)
            print('V2[mh,mw].shape', V2[mh,mw].shape)
            print('tensordot', np.tensordot(U2[mh,mw], V2[mh,mw], 1))
            #print('U2[mh,mw].dot(V2[mh,mw]).shape', U2[mh,mw].dot(V2[mh,mw]).shape)
            #print('U2[mh,mw].dot(V2[mh,mw])')
            #print(U2[mh,mw].dot(V2[mh,mw]))
            # M[:, :, :, mh, mw] = U2[mh,mw].dot(V2[mh,mw])
            M[:, :, :, mh, mw] = np.tensordot(U2[mh,mw], V2[mh,mw], 1)
            #Mtemp = np.zeros((Co, 1), dtype=np.float32)
            #for co in range(Co):
            #    sum = 0
            #    for ci in range(Ci):
            #       sum += U2[mh,mw,co,ci] * V2[mh,mw,ci,0]
            #    Mtemp[co,0] = sum
            #print('Mtemp', Mtemp)
    #print('M', M)
#    sys.exit(1)
    
    Mfull = M
    # inverse transform
    for co in range(Co):
        for th in range(tiles):
            for tw in range(tiles):
                O = Ofull[co,th * 4:(th+1)*4,tw*4:(tw+1)*4,n].reshape(4,4)
                M = Mfull[co, th, tw]
                Otmp = np.zeros((4, 6), dtype=np.float32)
                for i in range(6):
                    Otmp[0][i] = M[0][i] + M[1][i] + M[2][i] + M[3][i] + M[4][i]
                    Otmp[1][i] =         + M[1][i] - M[2][i] + 2 * M[3][i] - 2 * M[4][i]
                    Otmp[2][i] =         + M[1][i] + M[2][i] + 4 * M[3][i] + 4 * M[4][i]
                    Otmp[3][i] =         + M[1][i] - M[2][i] + 8 * M[3][i] - 8 * M[4][i] + M[5][i]
                #print('Otmp', Otmp)

                for i in range(4):
                    O[i][0] = Otmp[i][0] + Otmp[i][1] + Otmp[i][2] + Otmp[i][3] + Otmp[i][4]
                    O[i][1] =         + Otmp[i][1] - Otmp[i][2] + 2 * Otmp[i][3] - 2 * Otmp[i][4]
                    O[i][2] =         + Otmp[i][1] + Otmp[i][2] + 4 * Otmp[i][3] + 4 * Otmp[i][4]
                    O[i][3] =         + Otmp[i][1] - Otmp[i][2] + 8 * Otmp[i][3] - 8 * Otmp[i][4] + Otmp[i][5]
        #        O = O.reshape(Co, 4, 4, N)
                # print('AT.dot(M).dot(AT.T)', AT.dot(M).dot(AT.T))

def process(iH, iW, N, Ci, Co, kH=3, kW=3):
    np.random.seed(123)

    oH = iH
    oW = iW

    W = np.random.randn(Ci,kH,kW,Co).astype(np.float32)
    #W.fill(0)
    #W[0, 0, 0, 0] = 3
    #W[1, 0, 0, 0] = 4
    #W[1, 0, 1, 0] = 7
    #W[1, 1, 1, 0] = 0.3
    #W[0, 1, 1, 0] = 5
    #print('W', W)

    Wfull = W

    I = np.zeros((Ci,iH, iW,N), dtype=np.float32)
    I[:] = np.random.randn(*I.shape)
    #I.fill(0)
    #I[0, 0, 0, 0] = 5
    #I[0, 0, 1, 0] = 2
    #Inopadded = I
    Ifull = I

#    print('Inopadded', Inopadded)
#    print('Ipadded', Ipadded)

    print('Co', Co, 'iH', iH, 'iW', iW, 'N', N)
    O = np.zeros((Co, oH, oW, N), dtype=np.float32)
    Ofull = O

    for n in range(N):
        process_one(iH=iH, iW=iW, Ci=Ci, Co=Co, kH=kH, kW=kW, n=n, I=I, W=W, O=O)

    I = Ifull
    W = Wfull
    O = Ofull
    return {'W': W, 'O': O, 'I': I}

def simple1():
    image_size = 8
    N = 1
    Ci = 1
    Co = 1
 
    res = process(iH=image_size, iW=image_size, N=N, Ci=Ci,
        Co=Co)
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
        print('co', co, 'n', n)
        print('winograd')
        print(O[co,:,:,n].reshape(image_size, image_size))
        print('cpu')
        print(cpuO[co,:,:,n].reshape(image_size,image_size))
        assert np.allclose(O[co,:,:,n], cpuO[co,:,:,n], atol=1e-4)
    #printTensor(cpuO[0])

   # checkO(W=W, I=I, O=O, c=0, h=0, w=0, n=0)
   # checkO(W=W, I=I, O=O, c=0, h=0, w=0, n=1)
   # checkO(W=W, I=I, O=O, c=0, h=1, w=0, n=0)
   # checkO(W=W, I=I, O=O, c=0, h=0, w=1, n=0)
   # checkO(W=W, I=I, O=O, c=1, h=0, w=0, n=0)
#    checkO(W=W, I=I, O=O, c=3, h=2, w=1, n=27)

simple1()

