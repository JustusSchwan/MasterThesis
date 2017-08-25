from functools import partial

import cv2
import matplotlib
import numpy as np
import pylab
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
from skimage import measure
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import SpectralEmbedding
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
from sklearn import datasets
from math import sqrt

from dataflow import ports


class Clusters:
    def __init__(self, labels, n_clusters, data=None):
        self.centroids = np.zeros((n_clusters, 2))
        self.counts = np.zeros(n_clusters)
        self.std = np.zeros((n_clusters, 2))
        self.bounds = np.zeros((n_clusters, 4))
        for i in range(n_clusters):
            selection = \
                np.stack(np.where(labels == i + 1), axis=1) if data is None else data[labels == i, :2]
            self.counts[i] = selection.shape[0]
            self.centroids[i] = np.mean(selection, axis=0)[::-1]
            self.std[i, :] = np.std(selection, axis=0)[::-1]
            self.bounds[i, :2] = np.amin(selection, axis=0)[::-1]
            self.bounds[i, 2:] = np.amax(selection, axis=0)[::-1]


def _row_condition(val, cond):
    out = np.ones(val.shape[0], dtype=np.bool)
    if cond is not None:
        for i in range(val.shape[0]):
            if not cond(val[i, ...]):
                out[i] = False
    return out


def cluster_filter(clusters, condition_centroids=None, condition_counts=None, condition_std=None,
                   condition_bounds=None):
    if clusters is None:
        return None
    selection = np.bitwise_and(
        np.bitwise_and(_row_condition(clusters.centroids, condition_centroids),
                       _row_condition(clusters.counts, condition_counts)),
        np.bitwise_and(_row_condition(clusters.std, condition_std),
                       _row_condition(clusters.bounds, condition_bounds)))
    clusters.centroids = clusters.centroids[selection]
    clusters.counts = clusters.counts[selection]
    clusters.std = clusters.std[selection]
    clusters.bounds = clusters.bounds[selection]
    return clusters


def cluster_filter_op(condition_centroids=None, condition_counts=None, condition_std=None, condition_bounds=None):
    return partial(cluster_filter, condition_centroids=condition_centroids, condition_counts=condition_counts,
                   condition_std=condition_std, condition_bounds=condition_bounds)


def _cluster_scaler(clusters, scale):
    if clusters is None:
        return None
    clusters.centroids *= scale
    clusters.std *= scale
    clusters.counts *= scale
    clusters.bounds *= scale
    return clusters


def cluster_scaler(scale):
    return partial(_cluster_scaler, scale=scale)


class BlobClusteringConnectivity:
    def __init__(self, thresh_intensity=50, debug=False):
        self.sink_image = ports.StateSink()
        self.thresh_intensity = thresh_intensity
        self.debug = debug
        self.out_debug_image = ports.EventSource()

    def make_clusters(self):
        img = self.sink_image.get()
        if img is None:
            return None
        img = cv2.threshold(img, thresh=self.thresh_intensity, maxval=255, type=cv2.THRESH_BINARY)[1]

        labels, num = measure.label(img, connectivity=2, return_num=True)
        if num == 0:
            return None
        if self.debug:
            tmp = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
            cv2.applyColorMap(np.array(labels * 50, dtype=np.uint8), cv2.COLORMAP_AUTUMN, dst=tmp)
            tmp[labels == 0] = 0
            self.out_debug_image.fire(tmp)

        return Clusters(labels, n_clusters=num - 1)


class BlobClusteringDBSCAN:
    def __init__(self, dist, min_neighborhood=1, thresh_intensity=50, scale_intensity=1, debug=False):
        self.algo = DBSCAN(eps=dist, min_samples=min_neighborhood, n_jobs=-1)
        self.sink_image = ports.StateSink()
        self.thresh_intensity = thresh_intensity
        self.scale_intensity = scale_intensity
        self.debug = debug
        self.out_debug_image = ports.EventSource()
        self.data = None
        self.debug_buf = None

    def make_clusters(self):
        img = self.sink_image.get()
        if img is None:
            return None
        selection = np.where(img >= self.thresh_intensity)
        n_data = selection[0].shape[0]
        if self.data is None or self.data.shape[0] < n_data:
            self.data = np.zeros(shape=(n_data, 3), dtype=np.uint8)
        self.data[:n_data, :2] = np.stack(selection, axis=1)
        self.data[:n_data, 2] = img[selection] * self.scale_intensity
        clustering = self.algo.fit(self.data[:n_data, :])
        labels = clustering.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters_ == 0:
            return None

        if self.debug:
            if self.debug_buf is None or self.debug_buf.shape[:2] != img.shape[:2]:
                self.debug_buf = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
            offset = 50
            for i in range(n_clusters_):
                cluster = labels == i
                self.debug_buf[:, :, 0][self.data[:n_data, :][cluster, 0], self.data[:n_data, :][cluster, 1]] = int(
                    i * offset + offset)
            cv2.applyColorMap(self.debug_buf[:, :, 0], cv2.COLORMAP_AUTUMN, dst=self.debug_buf)
            self.debug_buf[img < self.thresh_intensity, :] = 0
            self.out_debug_image.fire(self.debug_buf)

        return Clusters(labels, n_clusters_, self.data[:n_data, :])


class _simple_clustering:
    def __init__(self, algo):
        self.sink_data = ports.StateSink()
        self.algo = algo

    def do_clustering(self):
        data = self.sink_data.get()
        if data is None:
            return None
        clustering = self.algo.fit(data)
        labels = clustering.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters_ == 0:
            return None

        return clustering


class SimpleDBSCAN(_simple_clustering):
    def __init__(self, dist, min_neighborhood=1):
        _simple_clustering.__init__(self, DBSCAN(eps=dist, min_samples=min_neighborhood, n_jobs=-1))


class SimpleMeanShift(_simple_clustering):
    def __init__(self, bandwidth=None):
        _simple_clustering.__init__(self, MeanShift(bandwidth=bandwidth, n_jobs=-1))


class SimpleAgglometrative(_simple_clustering):
    def __init__(self, n_clusters=2, *args, **kwargs):
        _simple_clustering.__init__(self, AgglomerativeClustering(n_clusters=n_clusters, *args, **kwargs))


class SilhouetteScore:
    def __init__(self, sample_size=5000, *args, **kwargs):
        self.sample_size = sample_size
        self.sink_data = ports.StateSink()
        self.score_args = args
        self.score_kwargs = kwargs

    def __call__(self, clusters):
        data = self.sink_data.get()
        if clusters is None or data is None:
            return None
        labels = clusters.labels_
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

        if n_clusters_ >= 2:
            return metrics.silhouette_score(data, labels, sample_size=2000, *self.score_args, **self.score_kwargs)
        else:
            return None


class ClusterPrinter:
    def __init__(self, num_images=20):
        # self.reducer = SpectralEmbedding()
        self.reducer = Isomap()
        self.sink_features = ports.StateSink()
        self.sink_filename = ports.StateSink()
        self.sink_image = ports.StateSink()
        self.num_images = num_images

    def __call__(self, clusters):
        features = self.sink_features.get()
        if clusters is None or features is None:
            return None
        valid = clusters.labels_ != -1
        view_data = features[valid]
        labels = clusters.labels_
        valid_labels = labels[valid]
        if len(valid_labels) == 0:
            return None
        choice = np.random.choice(range(len(valid_labels)), size=min(2000, len(valid_labels)), replace=False)
        view_data = self.reducer.fit(view_data[choice, :]).transform(features)
        print view_data.shape

        fig, ax = plt.subplots(figsize=(15, 15), dpi=300)

        num_clusters = len(set(valid_labels))
        patches = []
        for l in range(num_clusters):
            cluster = view_data[labels == l, :]
            try:
                hull = ConvexHull(cluster)
                patches.append(Polygon(cluster[hull.vertices, :]))
            except:
                pass
        p = PatchCollection(patches, cmap=matplotlib.cm.rainbow, alpha=0.4)
        ax.add_collection(p)

        invalid = np.invert(valid)
        plt.scatter(view_data[invalid, 0], view_data[invalid, 1], c='w', s=0.1)
        ax.set_facecolor('black')
        plt.scatter(view_data[valid, 0], view_data[valid, 1], c=valid_labels, s=0.1, cmap='rainbow')

        # Add a few images to the figure
        choices = []
        imgs_per_label = max(1, int(self.num_images / num_clusters))
        for l in range(num_clusters):
            cluster_ind = np.where(labels == l)[0]
            choices += np.random.choice(cluster_ind, size=min(imgs_per_label, len(cluster_ind)), replace=False).tolist()

        plt.scatter(view_data[choices, 0], view_data[choices, 1], c=labels[choices], s=180, marker='s',
                    cmap='rainbow')

        # Get the x and y data and transform it into pixel coordinates
        xy_pixels = ax.transData.transform(np.vstack([view_data[choices, 0], view_data[choices, 1]]).T)
        xpix, ypix = xy_pixels.T

        for i, c in enumerate(choices):
            img = self.sink_image.get(c)
            if img is None:
                continue
            scale = 50.0 / np.max(img.shape)
            img = cv2.cvtColor(cv2.resize(img, dsize=(0, 0), fx=scale, fy=scale), code=cv2.COLOR_BGR2RGB).astype(
                np.float32) / 255
            plt.figimage(img, xo=int(xpix[i]) - 25, yo=int(ypix[i]) - 25, zorder=10)

        pylab.savefig(self.sink_filename.get(), dpi=fig.dpi)
        plt.close('all')


class ClusterSamplePrinter:
    def __init__(self, num_images=20, num_rows=1):
        self.sink_filename = ports.StateSink()
        self.sink_image = ports.StateSink()
        self.num_images = num_images
        self.num_rows = num_rows

    def __call__(self, clusters):
        if clusters is None:
            return None
        valid_labels = clusters.labels_[clusters.labels_ != -1]

        num_clusters = len(set(valid_labels))

        # Add a few images to the figure
        choices = []
        for l in range(num_clusters):
            cluster_ind = np.where(clusters.labels_ == l)[0]
            choices.append(
                np.random.choice(cluster_ind, size=min(self.num_images*self.num_rows, len(cluster_ind)), replace=False).tolist())

        edge_length = 50
        spacing = 25

        out_img = np.zeros(shape=[edge_length * num_clusters * self.num_rows + (num_clusters - 1) * spacing,
                                  edge_length * self.num_images, 3],
                           dtype=np.uint8)

        print(out_img.shape)

        for i_cluster, cluster_choice in enumerate(choices):
            for i, c in enumerate(cluster_choice):
                img = self.sink_image.get(c)
                if img is None:
                    continue
                scale = float(edge_length) / np.max(img.shape)
                img = cv2.resize(img, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
                y_start = (i_cluster * self.num_rows + int(i / self.num_images)) * edge_length + i_cluster * spacing
                x_start = (i % self.num_images) * edge_length
                out_img[y_start:y_start + img.shape[0], x_start:x_start + +img.shape[1], :] = img

        cv2.imwrite(self.sink_filename.get(), out_img)


if __name__ == '__main__':
    cp = ClusterSamplePrinter(20, 2)
    digits = datasets.load_digits(n_class=6)
    X = digits.data


    class mock_clusters:
        def __init__(self, labels):
            self.labels_ = labels


    # cp.sink_features << (lambda: X)
    cp.sink_filename << (lambda: 'D:/tmp.png')
    cp.sink_image << (lambda _: cv2.imread('D:/Master Thesis/Other data/dlibfacepoints.png'))
    labels_ = digits.target
    labels_[10] = -1
    cp(mock_clusters(labels_))
