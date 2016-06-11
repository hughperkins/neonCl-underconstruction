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
Nervana's deep learning library
"""

try:
    from neon.version import VERSION as __version__  # noqa
except ImportError:
    import sys
    print("ERROR: Version information not found.  Ensure you have built "
          "the software.\n    From the top level dir issue: 'make'")
    sys.exit(1)


class NervanaObject(object):
    """
    Base (global) object available to all other classes.

    Args:
        name (str, optional)

    Attributes:
        be (Backend): Hardware backend being used.  See `backends` dir
        name (str, optional): The name assigned to a given instance.
    """
    be = None
    __counter = 0

    def __init__(self, name=None):
        if name is None:
            name = self.classnm + '_' + str(self.__counter)
        self.name = name
        self._desc = None
        type(self).__counter += 1

    @classmethod
    def gen_class(cls, pdict):
        return cls(**pdict)

    def __del__(self):
        type(self).__counter -= 1

    @property
    def classnm(self):
        """
        Convenience method for getting the class name
        """
        return self.__class__.__name__

