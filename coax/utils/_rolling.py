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
