import cPickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft

import expressivity


def iter_all(d_set):
    for _, vd in d_set.iteritems():
        for _, dt in vd.iteritems():
            for _, ft in dt.iteritems():
                yield ft


dumpfile = open('D:/Master Thesis/Dataset/features_raw.pkl', 'rb')

dataset = cPickle.load(dumpfile)


# mean/stdev normalization
def mean_stdev(d_set):
    m = np.zeros([1, 5])
    nelem = 0
    for elem in iter_all(d_set):
        m += elem[31:36]
        nelem += 1
    m /= nelem
    s = np.zeros([1, 5])
    for elem in iter_all(d_set):
        s += (elem[31:36] - m)*(elem[31:36] - m)
    s /= nelem-1
    s = np.sqrt(s)
    return m, s

mean, stdev = mean_stdev(dataset)

a, b = signal.butter(2, 0.1, btype='highpass')

# Number of sample points
N = 16
# sample spacing
T = 1
x = np.linspace(0.0, N*T, N)
xf = np.linspace(0.0, 1.0/(2.0*T), N/2)
plt.ion()

header = \
    "@relation weather" \
    "" \
    "@attribute temperature { sunny, overcast, rainy }" \
    "@attribute temperature numeric" \
    "@attribute humidity numeric" \
    "@attribute windy { TRUE, FALSE }" \
    "@attribute play { yes, no }" \

for person, videos in dataset.iteritems():
    for video, data in sorted(videos.iteritems()):
        next_frame = 0
        continuous = 0
        for frame, features in sorted(data.iteritems()):
            print ('{} {} {}'.format(person, video, frame))
            if next_frame != frame:
                continuous = 0
                continue
            next_frame = frame + 1
            continuous += 1
            ts[0:N-1, :] = ts[1:N, :]
            ts[N-1, :] = (features[31:36] - mean) / stdev

            if continuous > N-1:
                # print(ts)
                print(expressivity.analyze(ts[:, 0:2], 0.1))
                y = signal.lfilter(a, b, np.linalg.norm(ts[:, 2:5], ord=2, axis=1))
                yf = fft(y)
                plt.clf()
                plt.plot(xf, 2.0 / N * np.abs(yf[0:N / 2]))
                plt.pause(0.0005)
