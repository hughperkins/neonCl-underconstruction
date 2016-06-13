# Magic numbers and shift amounts for integer division
# Suitable for when nmax*magic fits in 32 bits
# Shamelessly pulled directly from:
# http://www.hackersdelight.org/hdcodetxt/magicgu.py.txt
def magic32(nmax, d):
    nc = ((nmax + 1) // d) * d - 1
    nbits = len(bin(nmax)) - 2
    for p in range(0, 2 * nbits + 1):
        if 2 ** p > nc * (d - 1 - (2 ** p - 1) % d):
            m = (2 ** p + d - 1 - (2 ** p - 1) % d) // d
            return (m, p)
    raise ValueError("Can't find magic number for division")


# Magic numbers and shift amounts for integer division
# Suitable for when nmax*magic fits in 64 bits and the shift
# lops off the lower 32 bits
def magic64(d):
    # 3 is a special case that only ends up in the high bits
    # if the nmax is 0xffffffff
    # we can't use 0xffffffff for all cases as some return a 33 bit
    # magic number
    nmax = 0xffffffff if d == 3 else 0x7fffffff
    magic, shift = magic32(nmax, d)
    if magic != 1:
        shift -= 32
    return (magic, shift)

def ceil_div(x, y):
    return -(-x // y)

