import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Normalizer

from sklearn.neighbors import kneighbors_graph

from scipy import stats
from sklearn.pipeline import Pipeline


class BagOfWordsModel:
    def __init__(self, n_clusters, k_neighbors, transforms=None):
        self.n_clusters = n_clusters
        self.k_neighbors = k_neighbors
        self.predictor = KNeighborsClassifier(n_neighbors=k_neighbors, algorithm='ball_tree', metric='euclidean', n_jobs=-1)

        normalizer_list = []
        if transforms is not None:
            normalizer_list += transforms
        normalizer_list.append(('normalize', Normalizer(norm='l2')))
        self.normalizer = Pipeline(normalizer_list)

    def fit(self, data):
        data = self.normalizer.fit_transform(data)
        cluster_labels = AgglomerativeClustering(n_clusters=self.n_clusters, linkage='ward', affinity='euclidean',
                                                 connectivity=kneighbors_graph(data, n_neighbors=2 * self.k_neighbors,
                                                                               include_self=False)).fit_predict(data)
        self.predictor.fit(data, cluster_labels)

    def predict(self, data):
        data = self.normalizer.transform(data)
        return self.predictor.predict(data)


def test_main():
    data = np.random.rand(1000, 2)
    model = BagOfWordsModel(3, 5)
    model.fit(data)
    print model.predict(np.random.rand(20, 2))


if __name__ == '__main__':
    test_main()
