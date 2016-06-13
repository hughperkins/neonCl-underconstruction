# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
"""
Defines gen_backend function
"""

#import atexit
import logging
import os
import sys
import numpy as np
from math import ceil

#from neon.backends.util import check_gpu
from neon.backends.nervanagpu import NervanaGPU


logger = logging.getLogger(__name__)

class make_backend(object):
    def __init__(self, datatype=np.float32,
                batch_size=0, stochastic_round=False, device_id=0,
                compat_mode=None,
                cache_dir=os.path.join(os.path.expanduser('~'), 'nervana/cache')):
        self.datatype = datatype
        self.batch_size = batch_size
        self.stochastic_round = stochastic_round
        self.device_id = device_id
        self.compat_mode = compat_mode
        self.cache_dir = cache_dir
        self.be = None
        self.be = NervanaGPU(default_dtype=self.datatype,
                        stochastic_round=self.stochastic_round,
                        device_id=self.device_id,
                        compat_mode=self.compat_mode,
                        cache_dir=self.cache_dir)

        self.be.bsz = self.batch_size
#        return self.be

    def __enter__(self):
        """
        Construct and return a backend instance of the appropriate type based on
        the arguments given. With no parameters, a single CPU core, float32
        backend is returned.

        Arguments:
            datatype (dtype): Default tensor data type. CPU backend supports np.float64, np.float32,
                              and np.float16; GPU backend supports np.float32 and np.float16.
            batch_size (int): Set the size the data batches.
            stochastic_round (int/bool, optional): Set this to True or an integer to implent
                                                   stochastic rounding. If this is False rounding will
                                                   be to nearest. If True will perform stochastic
                                                   rounding using default bit width. If set to an
                                                   integer will round to that number of bits.
                                                   Only affects the gpu backend.
            device_id (numeric, optional): Set this to a numeric value which can be used to select
                                           device on which to run the process
            compat_mode (str, optional): if this is set to 'caffe' then the conv and pooling
                                         layer output sizes will match that of caffe as will
                                         the dropout layer implementation
            deterministic (bool, optional): if set to true, all operations will be done deterministically.
            cache_dir (str, optional): a location for the backend to cache tuning parameters.

        Returns:
            Backend: newly constructed backend instance of the specifed type.

        Notes:
            * Attempts to construct a GPU instance without a CUDA capable card or without nervanagpu
              package installed will cause the program to display an error message and exit.
        """
        # init gpu
        return self.be

    def __exit__(self, *args):
        pass

