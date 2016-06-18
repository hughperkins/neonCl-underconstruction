# Multiplier and shift for integer division
# Suitable for when nmax*magic fits in 32 bits
# Shamelessly pulled directly from:
# http://www.hackersdelight.org/hdcodetxt/magicgu.py.txt  Doc from there:
""" Division by Multiplication Considered More Generally
   Handles arbitrary max dividends (not necessarily of the form 2**W -
1), and bignums.
   Computes the magic number for unsigned division, given a max value of
the dividend, and a divisor.
   This uses equations (26) and (27) on page 181, but with nc determined
from the maximum value of n, which is not necessarily 2**W - 1 for W =
the word size of n. It finds the smallest number that can serve for a
multiplier for all n in the range 0 <= n <= nmax.
   It is much simpler than the programs given in Hacker's Delight (e.g.,
that of Figure 10-2) because in Python, one need not be concerned about
overflowing the word size in which the computations are done.
  Example: magicgu(255, 7) is a 9-bit number, but magicgu(200, 7) is an
8-bit number.
   Also, magicgu(127, 7) is an 8-bit number, but magicgu(90, 7) is a
6-bit number. """
# Hugh says: Ok, the intuition behind this is:
# - we can divide by any power of 2, by bitshift
# - by multiplying by some number first, we can divide by other numbers, not just
#   powers of 2
# For example:
# - divide by 16 => we simply bit shift by 4
# - divide by 31 => we first multiply by 541202, then bitshift by 24
#    => this is the same as multiplying by (51202/pow(2,24))
#       which is multiplying by 0.0322582
#       which is dividing by 30.9999
#       since we are using integers, this rounds to division by 31 (this last line is hand-waving
#      more info eg in https://gmplib.org/~tege/divcnst-pldi94.pdf )
def get_div_mul_shift_32(nmax, d):
    nc = ((nmax + 1) // d) * d - 1
    nbits = len(bin(nmax)) - 2
    for p in range(0, 2 * nbits + 1):
        if 2 ** p > nc * (d - 1 - (2 ** p - 1) % d):
            m = (2 ** p + d - 1 - (2 ** p - 1) % d) // d
            return (m, p)
    raise ValueError("Can't find multiplier for dividing by %s" % d)


# Multiplier and shift for integer division
# Suitable for when nmax*magic fits in 64 bits and the shift
# lops off the lower 32 bits
def get_div_mul_shift_64(d):
    # 3 is a special case that only ends up in the high bits
    # if the nmax is 0xffffffff
    # we can't use 0xffffffff for all cases as some return a 33 bit
    # magic number
    nmax = 0xffffffff if d == 3 else 0x7fffffff
    magic, shift = get_div_mul_shift_32(nmax, d)
    if magic != 1:
        shift -= 32
    return (magic, shift)

def ceil_div(x, y):
    return -(-x // y)

