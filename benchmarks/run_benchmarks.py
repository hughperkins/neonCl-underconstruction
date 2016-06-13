"""
feed it a model name, will run benchmarks for that model
"""
import sys
import time
import importlib
import numpy as np
import argparse
from mycltensor import MyClTensor
from neon.layers.layer import Convolution
from neon.backends.make_backend import make_backend

parser = argparse.ArgumentParser()
parser.add_argument('--backend', default='cl')
parser.add_argument('--model', default='vgga')
args = parser.parse_args()

model_name = args.model
backend_name = args.backend
print('model_name', model_name, 'backend_name', backend_name)

its = 10

model = importlib.import_module('models.%s' % model_name)
backend = importlib.import_module('backends.%s' % backend_name)

batch_size = model.get_batchsize()
print('batch_size', batch_size)
for layer in model.get_net():
    if layer['Ci'] >= 4:
        print('RUNNING', layer)
        backend.test(batch_size, its, layer)
    else:
        print('SKIPPING', layer)

