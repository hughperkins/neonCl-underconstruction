# used as reference version, for comparison/correctness

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
                    #V = Vtmp.dot(BT.T)
                    #V2[:,:,ci,th,tw] = V
    #                for i in range(6):
     #                   for j in range(6):
      #                      V2[i, j, ci, th, tw] = V[i, j]
        timecheck('calced V2')
    return V2

