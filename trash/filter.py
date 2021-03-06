from scipy import signal
import numpy as np


class NpCircularArray:
    def __init__(self, n, length):
        self.arr = np.zeros([length, n])

    def append(self, arr):
        self.arr[0:-1, :] = self.arr[1:, :]
        self.arr[-1, :] = arr

    def set_all(self, arr):
        self.arr = np.tile(arr, (self.arr.shape[0], 1))

    def get(self):
        return self.arr


class ButterFilterArray:
    def __init__(self, n, n_filter, cutoff=0.5, filter_type='lowpass'):
        self.n = n
        self.b, self.a = signal.butter(n_filter, cutoff, btype=filter_type)
        self.zi = None
        self.arr = NpCircularArray(n_filter + 1, self.n)
        self.initialized = False

    def append_and_filter(self, arr):
        if self.initialized:
            self.arr.append(arr)
        else:
            self.arr.set_all(arr)
            zi = signal.lfilter_zi(self.b, self.a)
            self.zi = np.transpose(np.tile(zi, (self.n, 1))) * np.tile(self.arr.get()[0, :], (zi.shape[0], 1))
            self.initialized = True

        y, self.zi = signal.lfilter(self.b, self.a, self.arr.get(), axis=0, zi=self.zi)

        return y[-1, :]

    def reset(self):
        self.initialized = False

    def __call__(self, arr):
        return self.append_and_filter(arr)
