import numpy as np
import pyopencl as cl
from neoncl.backends.cuda_templates import _ew_types

# shuffles ABC => CBA
# grid is:  (ceil_div(C, 32), ceil_div(A, 32), B)
# block:    (32, 8, 1)
def get_shuffle_kernel_d3_cl(ctx, dtype):
    _shuffle_kernel = r"""
kernel void dimShuffle(
    global %(type)s* out, global const %(type)s* in,
    int BC, int C,
    int AB, int A)
{
    local %(type)s tile[32][33];

    int tx  = get_local_id(0);
    int ty  = get_local_id(1);
    int gx  = get_group_id(0);
    int gy  = get_group_id(1);
    int b = get_group_id(2);

    int c  = gx * 32 + tx;
    int a  = gy * 32 + ty;

    for (int j = 0; j < 32; j += 8)
    {
        int a_plus_j = a + j;
        if (a_plus_j < A && c < C)
            tile[ty + j][tx] = in[ a_plus_j*BC + b * C + c ];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    c = gx * 32 + ty;
    a = gy * 32 + tx;

    for (int i = 0; i < 32; i += 8)
    {
        int c_plus_i = c + i;
        if (c_plus_i < C && a < A)
            out[ c_plus_i*AB + b*A + a ] = tile[tx][ty + i];
    }
}
"""
    code = _shuffle_kernel % _ew_types[dtype]
    module = cl.Program(ctx, code).build()
    return module.dimShuffle

# shuffles CTRSK => KTRSC
# grid is:  (ceil_div(K, 32), ceil_div(C, 32), R*S*T)
# block:    (32, 8, 1)
def get_shuffle_kernel_cl(ctx, dtype):
    _shuffle_kernel = r"""
kernel void dimShuffle(
    global %(type)s* out, global const %(type)s* in,
    int TRSK, int RSK, int SK, int K,
    int TRSC, int RSC, int SC, int C,
    int RS, int T, int R, int S,
    int div_RS_mul, int div_RS_shift,
    int div_S_mul,  int div_S_shift)
{
    local %(type)s tile[32][33];

    int tx  = get_local_id(0);
    int ty  = get_local_id(1);
    int bk  = get_group_id(0);
    int bc  = get_group_id(1);
    int trs = get_group_id(2);

    int k  = bk * 32 + tx;
    int c  = bc * 32 + ty;

    int t = (trs * div_RS_mul) >> div_RS_shift;
    int rs = trs - t*RS;

    int r = (rs * div_S_mul) >> div_S_shift;
    int s = rs - r*S;

    for (int j = 0; j < 32; j += 8)
    {
        int cj = c + j;
        if (cj < C && k < K)
            tile[ty + j][tx] = in[ cj*TRSK + t*RSK + r*SK + s*K + k ];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    k = bk * 32 + ty;
    c = bc * 32 + tx;

    // Mirror RST
    s = S - s - 1;
    r = R - r - 1;
    t = T - t - 1;

    for (int i = 0; i < 32; i += 8)
    {
        int ki = k + i;
        if (ki < K && c < C)
            out[ ki*TRSC + t*RSC + r*SC + s*C + c ] = tile[tx][ty + i];
    }
}
"""
    code = _shuffle_kernel % _ew_types[dtype]
    module = cl.Program(ctx, code).build()
    return module.dimShuffle

