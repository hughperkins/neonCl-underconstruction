# ----------------------------------------------------------------------------
# Copyright 2015 Nervana Systems Inc.
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

import numpy as np

from neon import NervanaObject
from neon.backends.backend import Tensor


class Initializer(NervanaObject):
    """
    Abstract base class from which parameter tensor initializers inherit.
    """
    def fill(self, param):
        raise NotImplementedError()


class Gaussian(Initializer):
    """
    A class for initializing parameter tensors with values drawn from
    a normal distribution.

    Args:
        loc   (float, optional): The mean of the normal (mu).
        scale (float, optional): The standard deviation of the normal (sigma).
    """
    def __init__(self, loc=0.0, scale=1.0, name="gaussianInit"):
        super(Gaussian, self).__init__(name=name)
        self.loc, self.scale = (loc, scale)

    def fill(self, param):
        param[:] = self.be.rng.normal(self.loc, self.scale, param.shape)


