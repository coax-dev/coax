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

from collections import deque


class RollingAverage:
    def __init__(self, n=100):
        self._value = 0.
        self._deque = deque(maxlen=n)

    @property
    def value(self):
        return self._value

    def update(self, observed_value):
        if len(self._deque) == self._deque.maxlen:
            self._value += (observed_value - self._deque.popleft()) / self._deque.maxlen
            self._deque.append(observed_value)
        else:
            self._deque.append(observed_value)
            self._value += (observed_value - self._value) / len(self._deque)
        return self._value


class ExponentialAverage:
    def __init__(self, n=100):
        self._value = 0.
        self._len = 0
        self._maxlen = n

    @property
    def value(self):
        return self._value

    def update(self, observed_value):
        if self._len < self._maxlen:
            self._len += 1
        self._value += (observed_value - self._value) / self._len
        return self._value
