from winogradcl.util import math_helper

def test1():
    a = 164543
    b = 31
    print('a // b', a // b)
    mul, shift = math_helper.get_div_mul_shift_32(1000000, 31)
    print('mul', mul, 'shift', shift)
    print('(a * mul) >> shift)', (a * mul) >> shift)

if __name__ == '__main__':
    test1()

