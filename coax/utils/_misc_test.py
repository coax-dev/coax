# ------------------------------------------------------------------------------------------------ #
# MIT License                                                                                      #
#                                                                                                  #
# Copyright (c) 2020, Microsoft Corporation                                                        #
#                                                                                                  #
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software    #
# and associated documentation files (the "Software"), to deal in the Software without             #
# restriction, including without limitation the rights to use, copy, modify, merge, publish,       #
# distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the    #
# Software is furnished to do so, subject to the following conditions:                             #
#                                                                                                  #
# The above copyright notice and this permission notice shall be included in all copies or         #
# substantial portions of the Software.                                                            #
#                                                                                                  #
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING    #
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND       #
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,     #
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,   #
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.          #
# ------------------------------------------------------------------------------------------------ #

import os

import pytest
import gym

from .._base.test_case import TestCase, MemoryProfiler
from ._misc import StrippedEnv, reload_recursive


class TestMiscUtils(TestCase):

    @pytest.mark.skipif(os.environ.get('CI') == 'true', reason="too unreliable for automation")
    def test_stripped_env_memory_footprint(self):
        reload_recursive(gym)

        def get_env():
            env = gym.make('MsPacman-v0')
            env.reset()
            return env

        def get_stripped_env():
            env = get_env()
            return StrippedEnv(env)

        with MemoryProfiler() as mp1:
            env = get_env()
        del env

        with MemoryProfiler() as mp2:
            env = get_stripped_env()
        del env

        # stripped env should have a significantly reduced memory footprint
        self.assertGreater(mp1.memory_used, 17 * mp2.memory_used)
