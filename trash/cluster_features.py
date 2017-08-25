import csv
import os
import shutil

import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from scipy.spatial import ConvexHull

from matplotlib import pyplot as plt
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

import pylab

from scipy.ndimage.filters import gaussian_filter

def softmax(v):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(v - np.max(v))
    return e_x / e_x.sum()


loadfile = open('D:/Master Thesis/Dataset/pos_features.csv', 'rb')

reader = csv.reader(loadfile, delimiter=',')

n_data = 0
n_features = 0
names = {}
name = None
line_num = 0

feature_desc = ()

# Get data size and lines for each individual
for line_num, l in enumerate(reader):
    if line_num == 0:
        feature_desc = l[:-2]
        continue
    n_data += 1
    n_features = len(l) - 2
    if name is None:
        name = l[n_features].split('_')[0]
        names[name] = [line_num - 1, 0]
    elif name != l[n_features].split('_')[0]:
        names[name][1] = line_num - 1
        name = l[n_features].split('_')[0]
        names[name] = [line_num - 1, 0]

names[name][1] = line_num - 1

print (names)

dataset = np.zeros([n_data, n_features])

loadfile.seek(0)
for i, l in enumerate(reader):
    if i == 0:
        continue
    if i == n_data:
        break
    dataset[i - 1, :] = np.asarray(l[:-2])

# for n_kmeans in range(2, 15):
removed_entries = []
candidates = []
removed_entries.append([[], 0])
candidates.append(set())
for i in range(n_features):
    removed_entries.append([[i], 0])
    candidates.append(set([i]))

i_removed = 0
while True:
    selection = np.array([x not in removed_entries[i_removed][0] for x in range(n_features)], dtype=np.bool)
    print(selection)
    overall_score = 0
    successful_ops = 0
    fails = 0

    output_dir = 'D:/Master Thesis/Clustering/tmp2'

    excluded_string = '-'.join(np.array(feature_desc, dtype=np.str)[np.invert(selection)].tolist())
    included_string = '-'.join(np.array(feature_desc, dtype=np.str)[selection].tolist())

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for name, lines in names.iteritems():
        print(name)

        start = lines[0]
        end = lines[1]
        step = 1
        if end - start > 5000:
            step = int((end - start) / 5000)

        data = dataset[start:end:step, selection]

        data = StandardScaler().fit_transform(data)

        num_samples = 10
        nbrs = NearestNeighbors(n_neighbors=num_samples, metric='euclidean', algorithm='ball_tree').fit(data)
        distances, _ = nbrs.kneighbors(data)
        nn_distances = gaussian_filter(np.sort(distances[:, -1]), sigma=5)
        grad_distances = np.gradient(nn_distances)
        grad2_distances = np.gradient(grad_distances)
        eps = nn_distances[np.argmax(grad_distances[np.logical_and(grad_distances < 0.002, grad2_distances < 0.002)])]
        print(eps)

        plt.plot(nn_distances, 'r-', np.gradient(nn_distances), 'b-', grad2_distances, 'g-')
        plt.show()

        db = DBSCAN(eps=eps, min_samples=num_samples).fit(data)
        # core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        # core_samples_mask[db.core_sample_indices_] = True
        # db = KMeans(n_clusters=n_kmeans, algorithm='elkan').fit(data)
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

        title_string = 'Clusters: ' + str(n_clusters_)
        print('Estimated number of clusters: %d' % n_clusters_)
        if n_clusters_ >= 2:
            score = metrics.silhouette_score(data, labels, sample_size=5000)
            successful_ops += 1
            overall_score += score
            print("Silhouette Coefficient: %0.3f" % score)
            title_string += ', Score: ' + str(score)
        else:
            fails += 1
            title_string += ', Score: NA'

        #####################################################
        #                    PLOTTING                       #
        #####################################################
        fig, ax = plt.subplots(figsize=(8, 7), dpi=100)
        pca = PCA(n_components=2)
        pca.fit(data)
        data = pca.transform(data)

        patches = []

        for l in range(n_clusters_):
            cluster = data[labels == l, :]
            try:
                hull = ConvexHull(cluster)
                patches.append(Polygon(cluster[hull.vertices, :]))
            except:
                pass
                # plt.plot(cluster[hull.vertices, 0], cluster[hull.vertices, 1], 'r-', lw=1)
                # plt.plot(cluster[hull.vertices[-1:0], 0], cluster[hull.vertices[-1:0], 1], 'r-', lw=1)

        p = PatchCollection(patches, cmap=matplotlib.cm.rainbow, alpha=0.4)

        ax.add_collection(p)

        ax.set_facecolor('black')
        plt.scatter(data[labels != -1, 0], data[labels != -1, 1], c=labels[labels != -1], s=1, cmap='rainbow')

        ax.set_title(name + '\n' + 'Excluded: ' + excluded_string + '\n' + title_string)
        pylab.savefig('{}/{}.png'.format(output_dir, name), bbox_inches='tight')
        plt.close('all')
        # plt.show()

    if fails > 5:
        overall_score = -1
    else:
        overall_score /= successful_ops
    removed_entries[i_removed][1] = overall_score
    print '{}, {}'.format(i_removed, removed_entries[i_removed][:])
    i_removed += 1
    if i_removed == len(removed_entries):
        scores = np.array([removed_entries[i][1] for i in range(len(removed_entries))])
        new_item = set()
        while new_item in candidates or len(new_item) > n_features - 2:
            s = np.random.choice(len(removed_entries), size=2, replace=False, p=softmax(scores))
            new_item = set(removed_entries[s[0]][0] + removed_entries[s[1]][0])
        removed_entries.append([list(new_item), 0])
        candidates.append(new_item)

    move_dir = 'D:/Master Thesis/Clustering/dbschan_eps_{}_samples_10/{}-include-{}-exclude-{}'.format(
        str(eps_dbscan).replace(
            '.', '_'),
        str(overall_score).replace(
            '.', '_'),
        included_string,
        excluded_string)
    if os.path.exists(move_dir):
        shutil.rmtree(move_dir)

    shutil.move(output_dir, move_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if len(removed_entries) == 50:
        break
