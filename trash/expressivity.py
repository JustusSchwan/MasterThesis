import numpy as np
from scipy.spatial.distance import pdist, squareform


def extent(series):
    return np.nanmax(squareform(pdist(series)))


def analyze(series, dist=1):
    spatial_extent = extent(series)

    v = np.gradient(series, dist, axis=0)

    speed = np.sum(np.linalg.norm(series[:-1] - series[1:], axis=1))
    energy_col = np.amax(np.abs(v), axis=0)
    energy = np.linalg.norm(energy_col, ord=2)
    fluidity = np.linalg.norm(np.std(v, axis=0), ord=2)

    return np.concatenate(
        (np.array([spatial_extent]), np.array([speed]), energy_col, np.array([energy]), np.array([fluidity])),
        axis=0)


def description(n, prefix=""):
    desc = ['{}spatial_ext'.format(prefix), "{}speed".format(prefix)]
    for i in range(n):
        desc.append("{}energy_col{}".format(prefix, i))
    desc.append("{}energy".format(prefix))
    desc.append("{}fluidity".format(prefix))
    return desc
