import numpy as np
from scipy.spatial.distance import pdist, squareform


def extent(series):
    return np.nanmax(squareform(pdist(series)))


def analyze(series, dist=1):
    # extent(series)
    spatial_extent = np.sum(np.amax(series, axis=0) - np.amin(series, axis=0))

    v_abs = np.linalg.norm(np.gradient(series, dist, axis=0), ord=2, axis=1)

    speed = np.mean(v_abs)
    fluidity = np.std(v_abs)

    return np.array([spatial_extent, speed, fluidity])


def description(prefix=""):
    return ['{}spatial_ext'.format(prefix), "{}speed".format(prefix), '{}fluidity'.format(prefix)]
