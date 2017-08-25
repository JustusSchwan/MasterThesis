import numpy as np
from dataflow import ports


class NpCircularArray:
    def __init__(self, n, length):
        self.arr = np.zeros([length, n])
        self.capacity = length
        self.len = 0

    def append(self, arr):
        self.arr[0:-1, :] = self.arr[1:, :]
        self.arr[-1, :] = arr
        if self.len < self.capacity:
            self.len += 1

    def set_all(self, arr):
        self.arr = np.tile(arr, (self.arr.shape[0], 1))

    def get(self):
        return self.arr

    def get_cropped(self):
        return self.arr[-self.len:, :]

    def set_cropped(self, arr):
        self.arr[-self.len:, :] = arr

    def __len__(self):
        return self.len

    def clear(self):
        self.len = 0


class AutoCircularArray:
    def __init__(self, n, length):
        self.arr = NpCircularArray(n, length)
        self.source_buf = ports.EventSource()

    def __call__(self, data):
        self.arr.append(data)
        if len(self.arr) == self.arr.capacity:
            self.source_buf.fire(self.arr.get())
            return self.arr.get()
        return None

    def reset_state(self):
        self.arr.clear()


class GrowingNumpyArray:
    def __init__(self):
        self.arr = None
        self.n = 0

    def __call__(self):
        if self.arr is None:
            return None
        return self.arr[:self.n, :]

    def __len__(self):
        return self.n

    def append(self, vec):
        if vec is None:
            return
        if self.arr is None:
            self.arr = np.zeros((1, len(vec)))
        if self.arr.shape[0] <= self.n:
            self.arr.resize((int(max(6, 1.5 * self.n)), self.arr.shape[1]))
        self.arr[self.n, :] = vec[:]
        self.n += 1

    def clear(self):
        self.n = 0
