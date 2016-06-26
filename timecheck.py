import time

last = 0
def inittime():
    global last
    last = time.time()

def timecheck(label):
    global last
    now = time.time()
    print(label, '%.2f ms' % ((now - last) * 1000))
    last = now

