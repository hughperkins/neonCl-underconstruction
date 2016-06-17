import time
import numpy as np
import pyopencl as cl
from winogradcl.backends.cuda_templates import _ew_types
from winogradcl.backends.kernels.cl.callkernel import call_cl_kernel


mf = cl.mem_flags

class ShuffleRunner(object):
    def __init__(self, ctx, q, dtype):
        self.ctx = ctx
        self.q = q
        self.dtype = dtype
        self.shuffle_kernel_cl = _get_shuffle_kernel_cl(self.ctx, dtype.str[1:])
        
    def execute(self, *args):
        call_cl_kernel(self.shuffle_kernel_cl,
            self.q, *args
        )


def _get_shuffle_kernel_cl(ctx, dtype):
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

    int t = (t * div_RS_mul) >> div_RS_shift;
    int rs = trs - t*RS;

    int r = (r * div_S_mul) >> div_S_shift;
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

