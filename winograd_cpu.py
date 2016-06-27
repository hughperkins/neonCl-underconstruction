# used as reference version, for comparison/correctness

import numpy as np
from timecheck import inittime, timecheck
from neoncl.util import math_helper


def calcU(W):
    Ci = W.shape[0]
    kH = W.shape[1]
    kW = W.shape[2]
    Co = W.shape[3]

    G = np.array([[1/4,0,0],
        [-1/6,-1/6,-1/6],
        [-1/6,1/6,-1/6],
        [1/24,1/12,1/6],
        [1/24,-1/12,1/6],
        [0,0,1]], dtype=np.float32)

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
    timecheck('calced U2')

    # layout:
    # [xi, nu, co, ci]

    return U2

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

    BT = np.array([[4,0,-5,0,1,0],
          [0,-4,-4,1,1,0],
          [0,4,-4,-1,1,0],
          [0,-2,-1,2,1,0],
          [0,2,-1,-2,1,0],
          [0,4,0,-5,0,1]], dtype=np.float32)

    V2 = np.zeros((N, 6, 6, Ci, tiles, tiles), dtype=np.float32)
    timecheck('allocaed V2')
    for n in range(N):
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
                    V2[n, :,:,ci,th,tw] = Vtmp.dot(BT.T)
    timecheck('calced V')
    return V2

def calcM(N, Co, U, V):
    GK = U.shape[2]
    Ci = U.shape[3]
    tiles = V.shape[3]
    GN = V.shape[2]

    U = U.transpose(0,1,2,4,3).reshape(6,6,GK * 32,Ci)[:,:,:Co,:]

    V = V.transpose(
        2,6,0,1,5,3,4).reshape(
        GN * 32, 6, 6, Ci, tiles, tiles)[:N]

    M = np.zeros((N, Co, tiles, tiles, 6, 6), dtype=np.float32)
    for n in range(N):
        for xi in range(6):
            for nu in range(6):
                M[n,:, :, :, xi, nu] = np.tensordot(U[xi,nu], V[n,xi,nu], 1)
    timecheck('calced M')
    return M

def calcM_blocked_l2(U, V, axes):
    R1 = np.tensordot(U, V, axes)
    return R1

def calcM_blocked_l1(N, Co, U, V):
    GK   = U.shape[2]
    Ci = U.shape[3]
    tiles = V.shape[3]
    GN = V.shape[2]

    M = np.zeros((GN, 32, GK, 32, tiles, tiles, 6, 6), dtype=np.float32)

    # new layouts:
    # U
    # [xi, nu, co // 32,         ci, co % 32]
    # V
    # [xi, nu,  n // 32, th, tw, ci,  n % 32]

    # each block:
    # U [ci, co % 32]
    # V [ci, ni % 32]

    N_blocksize = 32
    ci_blocksize = 32
    Co_blocksize = 32
    printed_size = False
    for Co_block in range(GK):
        U_block = U[:,:,Co_block]
        for N_block in range(GN):
            for th in range(tiles):
                for tw in range(tiles):
                    V_block = V[:, :, N_block, th, tw]
                    M_block = M[N_block, :, Co_block, :, th, tw]
                    for mh in range(6):
                        for mw in range(6):
                           left = U_block[mh,mw]
                           right = V_block[mh,mw]
                           if not printed_size:
                               printed_size = True
                               print('left.shape', left.shape, 'right.shape', right.shape)
                           src = calcM_blocked_l2(left, right, ([0], [0]))
                           dst = M_block[:, :, mh, mw]
                           dst[:] = src.T
    M = M.reshape(GN * 32, GK * 32, tiles, tiles, 6, 6)
    M = M[:N, :Co]
    timecheck('calced M')
    return M

def calcO(M):
    N = M.shape[0]
    Co = M.shape[1]
    tiles = M.shape[2]
    oH = tiles * 4  # is this always true?  anyway, it's true for now...
    oW = tiles * 4

    O = np.zeros((Co, oH, oW, N), dtype=np.float32)
    Mfull = M
    Ofull = O

    AT = np.array([[1,1,1,1,1,0],
        [0,1,-1,2,-2,0],
        [0,1,1,4,4,0],
        [0,1,-1,8,-8,1]], dtype=np.float32)

    timecheck('allocated AT')

    # inverse transform
    Otmp = np.zeros((4, 6), dtype=np.float32)
    for n in range(N):
        for co in range(Co):
            for th in range(tiles):
                for tw in range(tiles):
                    O = Ofull[co,th * 4:(th+1)*4,tw*4:(tw+1)*4,n]
                    M = Mfull[n, co, th, tw]
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
    return Ofull

