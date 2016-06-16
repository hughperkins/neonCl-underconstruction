"""
api for running convolutions using winograd

status: in progress

approximate guidelines/requirements:
- caller should handle opencl context and queue setup
- caller should allocate cl buffers
- library can/should provide a means to provide required dimensions of buffers to caller
- library will check dimensions of incoming buffers
"""

def fprop(ctx, queue, I, I_layout, W, W_layout, O, O_layout):
    """
    layout should be a permutation of letters:
    - for I, letters:  'C H W N' or 'N H W C'
    - for W, letters:  'Ci H W Co' or 'Co H W Ci'
    """


