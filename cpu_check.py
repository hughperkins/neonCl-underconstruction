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

