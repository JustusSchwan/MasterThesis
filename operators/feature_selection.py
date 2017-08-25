from sklearn import svm
from sklearn import feature_selection as fs
from sklearn.cluster import FeatureAgglomeration
from dataflow import ports
import numpy as np


class SVMFeatureSelector:
    def __init__(self, C=1):
        self.model = svm.LinearSVC(C=1, penalty="l1", dual=False)
        self.dirty = True

    def _fit(self, features, labels):
        if features is None or labels is None:
            self.dirty = True
            return
        if features.shape[0] == 0 or labels.shape[0] == 0 or len(np.unique(labels)) < 2:
            self.dirty = True
            return
        self.model.fit(features, np.ravel(labels))
        self.dirty = False

    def get_feature_selection(self, features, labels):
        self._fit(features, labels)
        if self.dirty:
            return None
        return fs.SelectFromModel(self.model, prefit=True).get_support()


class AgglomerativeFeatureTransformer:
    def __init__(self, n_clusters=2):
        self.model = FeatureAgglomeration(n_clusters=n_clusters)

    def __call__(self, data):
        if data is None or data.shape[0] == 0:
            return None
        return self.model.fit_transform(data)


class Filter:
    def __init__(self):
        self.sink_selection = ports.StateSink()

    def __call__(self, data):
        selection = self.sink_selection.get()
        if selection is None:
            return None
        return data[:, selection]
