from __future__ import print_function

import cPickle
import itertools
import sqlite3
from os import path

import matplotlib.pyplot as plt
import numpy as np
import pylab
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.svm import SVR

import BagOfWordsModel
from dataflow import connectables
from dataflow import logging
from util.np_buffers import GrowingNumpyArray

project_folder = 'D:/Master Thesis/'
logfiles_folder = 'D:/Master Thesis/logfiles/'
source_folder = 'C:/Master Thesis/Dataset/'
cache_folder = 'D:/Master Thesis/BOWTransformed'


# Gets features (expreeivity, geometric) and boredom, engagement, frustration annotations for the given user
def getFeaturesAndAnnotations(user):
    db = sqlite3.connect(path.join(source_folder, 'Users.sqlite'))

    c = db.cursor()
    c.execute(
        'SELECT sessionId, Boredom_Annotation, Engagement_Annotation, Frustration_Annotation FROM StudentsSession'
        ' WHERE userId=?', (user,))
    annotations = np.array(list(session for session in c), dtype=np.int32)

    sessions = annotations[:, 0]

    if len(sessions) == 0:
        return None

    paths = list(path.join(logfiles_folder, '{}_{}.features'.format(user, session)) for session in sessions)
    for vid_path in paths:
        assert path.exists(vid_path), vid_path + ' does not exist'

    db.close()

    def get_features(filename):
        log_reader = logging.Logger(logging.LoggingMode.REPLAY)

        store = GrowingNumpyArray()
        source_features = connectables.StateToEvent(log_reader.source_state('features'))
        source_features.source >> store.append

        log_reader.open(filename)

        while True:
            if not log_reader.read():
                break
            source_features.get_and_fire()
            log_reader.write()

        return store()

    session_features = list(get_features(vid_path) for vid_path in paths)

    not_none_sessions = list(x is not None for x in session_features)
    session_features = list(x for x, b in zip(session_features, not_none_sessions) if b)

    labels = annotations[np.array(not_none_sessions), 1:]

    return session_features, labels


# Returns features and continuous labels for all sessions of the given user
# session_features is a set of videos. There are several frames in each video, each of which has a feature vector
# Each session has one label for each of boredem, engagement and frustration
# A session is considered valid when features exist for at least 100 frames
def getValidSessions(user):
    session_features, annotations = getFeaturesAndAnnotations(user)

    valid_sessions = list(x.shape[0] > 100 for x in session_features)
    session_features = list(x for x, b in zip(session_features, valid_sessions) if b)

    labels = annotations[np.array(valid_sessions), :]

    assert (len(session_features) == len(labels))
    return session_features, labels


# Returns features and class labels for all sessions of the given user
# session_features is a set of videos. There are several frames in each video, each of which has a feature vector
# Each session is labeled with either 'bored', 'engaged' or 'frustrated'
# A session is considered valid when features exist for at least 100 frames and the max continuous label is unique
# e.g. a session with boredom 2, engagement 4 and frustration 4 is not valid and therefore excluded
def getValidDiscretizedSessions(user):
    session_features, annotations = getFeaturesAndAnnotations(user)

    valid_sessions = np.logical_and(
        np.array(list(x is not None and x.shape[0] > 100 for x in session_features), dtype=np.bool_),
        np.sum(annotations.T == annotations.max(axis=1), axis=0) == 1)

    session_features = list(x for x, b in zip(session_features, valid_sessions) if b)
    labels = np.argmax(annotations[valid_sessions], axis=1)

    assert (len(session_features) == len(labels))
    return session_features, labels


# Constructs a bag of words model which uses agglomerative clustering with pseudo cosine distance
# if pca_components is >0, a PCA transformer with the respective number of components will be included in the pipeline
# Data comes from all users
def constructBowModel(n_clusters, pca_components):
    user_list = list(range(1, 34))

    temp_file = path.join(cache_folder, '/bow_model_cl_{}_train_{}{}.pkl'.
                          format(n_clusters,
                                 str(user_list).replace(',', '_')[1:-1],
                                 '_with_pca_' + str(pca_components) if pca_components > 0 else ''))

    if path.exists(temp_file):
        with open(temp_file, 'rb') as f:
            return cPickle.load(f)
    else:
        user_sessions = []
        for u in user_list:
            sessions = getFeaturesAndAnnotations(u)
            if sessions is not None and len(sessions[0]) > 0:
                user_sessions.append(np.vstack(sessions[0]))

        normalizer = [('Scaler', MinMaxScaler())]
        if pca_components > 0:
            normalizer.append(('PCA', PCA(n_components=pca_components)))

        model = BagOfWordsModel.BagOfWordsModel(n_clusters=n_clusters, k_neighbors=5, transforms=normalizer)
        model.fit(np.vstack(user_sessions))
        with open(temp_file, 'wb') as f:
            cPickle.dump(model, file=f, protocol=-1)
        return model


# Utility function, returns features for given user transformed with the BoW model as described in constructBowModel
# session_getter is a function pointer to either getValidSessions or getValidDiscretizedSessions
# model_cache is used to cache the BoW model to prevent frequent expensive fitting
def bowTransformedFeatures(user, n_clusters, pca_components, session_getter, model_cache):
    temp_file = path.join(cache_folder, 'usr_{}_cl_{}_{}{}.pkl'.
                          format(user, n_clusters, session_getter.__name__,
                                 '_with_pca_' + str(pca_components) if pca_components > 0 else ''))

    if path.exists(temp_file):
        with open(temp_file, 'rb') as f:
            return cPickle.load(f)
    else:
        if model_cache is None:
            model_cache = constructBowModel(n_clusters, pca_components)

        sessions = session_getter(user)
        if sessions is None or len(sessions[1]) == 0:
            with open(temp_file, 'wb') as f:
                cPickle.dump(None, file=f, protocol=-1)
            return None

        sessions_features, labels = sessions
        transformed_features = []
        for features in sessions_features:
            prediction = model_cache.predict(np.vstack(features))
            result = np.histogram(prediction, bins=np.arange(n_clusters + 1), normed=True)[0]
            transformed_features.append(result)
        assert (len(transformed_features) == len(labels))

        with open(temp_file, 'wb') as f:
            cPickle.dump([np.array(transformed_features), labels], file=f, protocol=-1)

        return [np.array(transformed_features), labels]


# Computes cross validation for regressing the affect labels
def CrossValRegression(features, labels, **params):
    loo = LeaveOneOut()
    reg = SVR(epsilon=0.1, **params)

    result = []
    for i in range(3):
        result.append(
            -1 * cross_val_score(reg, X=features, y=labels[:, i], cv=loo, scoring='neg_mean_absolute_error'))
    return np.array(result).T


# computes predictions of a weak regressor averaging the affect labels per candidate
def getMeanRegression(user):
    temp_file = path.join(cache_folder, 'mean_regression_usr_{}.pkl'.format(user))
    if path.exists(temp_file):
        with open(temp_file, 'rb') as f:
            return cPickle.load(f)

    session_data = getValidSessions(user)
    if session_data is None:
        return None
    session_features, labels = session_data

    if len(labels) == 0:
        return None

    err = []
    for i in range(labels.shape[0]):
        others = np.delete(labels, [i], axis=0)
        assert others.shape[1] == 3
        err.append(np.abs(labels[i, :] - np.mean(others, axis=0)))

    ret = np.array(err)
    with open(temp_file, 'wb') as f:
        cPickle.dump(ret, f, -1)
    return ret


# Dumps a matrix to console in csv format
def print_csv(mat):
    for l1 in mat:
        for l2 in l1:
            print(l2, ',', sep='', end='')
        print()


# Predicts affect labels using a Support Vector Regressor and compares them to the weak predictor
def EvaluateRegression(users, n_clusters, pca_components):
    comp_data = []
    model_cache = None

    features = []
    labels = []

    for user in users:
        session_data = bowTransformedFeatures(user, n_clusters, pca_components, getValidSessions, model_cache)
        if session_data is not None and len(session_data[0]) > 1:
            features.append(session_data[0])
            labels.append(session_data[1])
            comp_data.append(getMeanRegression(user))

    user_data = CrossValRegression(np.vstack(features), np.vstack(labels), C=1)
    comp_data = np.vstack(comp_data)

    csv_data = np.zeros((3, 3))

    for i in range(3):
        p = stats.ttest_rel(user_data[:, i], comp_data[:, i]).pvalue
        csv_data[i, :] = [np.mean(user_data[:, i]), np.mean(comp_data[:, i] - user_data[:, i]), p]
    print_csv(csv_data)


# Predicts affect labels per candidate using a Support Vector Regressor and compares them to the weak predictor
def EvaluateRegressionPerCandidate(users, n_clusters, pca_components):
    comp_data_all = []
    model_cache = None

    features = []
    labels = []

    chosen_ones = []

    for user in users:
        session_data = bowTransformedFeatures(user, n_clusters, pca_components, getValidSessions, model_cache)
        if session_data is not None and len(session_data[0]) > 1:
            features.append(session_data[0])
            labels.append(session_data[1])
            comp_data_all.append(getMeanRegression(user))
            chosen_ones.append(user)

    csv_data = np.zeros((3, 3))

    label = ['Boredom', 'Engagement', 'Frustration']

    num_executed = 0

    for f, l, comp_data, chosen_one, in zip(features, labels, comp_data_all, chosen_ones):
        if len(labels) < 10 or not all(np.mean(comp_data, axis=0).tolist()):
            continue
        num_executed += 1
        user_data = CrossValRegression(f, l, C=1)

        for i in range(3):
            p = stats.ttest_rel(user_data[:, i], comp_data[:, i]).pvalue
            csv_data[i, :] = [np.mean(user_data[:, i]), np.mean(comp_data[:, i] - user_data[:, i]), p]

        for i in range(3):
            print((chosen_one if i == 0 else ''), ',', sep='', end='')
            print(label[i], *(csv_data[i, :].tolist()), sep=',')


# Returns BoW transformed features and affect class labels
def GetDiscretizedFeaturesAndLabels(users, n_clusters, pca_components):
    model = None
    features = []
    labels = []
    for user in users:
        data = bowTransformedFeatures(user, n_clusters, pca_components, getValidDiscretizedSessions, model)
        if data is not None and len(data[0]) > 0:
            features.append(data[0])
            labels.append(data[1])

    labels = np.hstack(labels)
    print(labels)
    print(len(labels))
    features = np.vstack(features)
    return features, labels


# plots a confusion matrix to the current pyplot axes object
# largely taken from http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.gca().set_title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")


# Predicts affect labels and estimates performance using LOO cross validation
# Takes a list of n_clusters and a list of C to make a matrix of confusion matrices
def MakeConfusionMatrix(users, n_clusters_list, pca_components, c_list):
    plt_size = 2.5
    fig, axes = plt.subplots(nrows=len(n_clusters_list), ncols=len(c_list),
                             figsize=(plt_size * len(c_list), plt_size * len(n_clusters_list)), dpi=300)
    np.set_printoptions(precision=2)

    for i, n_clusters in enumerate(n_clusters_list):
        for j, c in enumerate(c_list):
            features, labels = GetDiscretizedFeaturesAndLabels(users, n_clusters, pca_components)
            loo = LeaveOneOut()
            classifier = SVC(C=c, class_weight='balanced')

            predicted = cross_val_predict(classifier, X=features, y=labels, cv=loo)
            mat = confusion_matrix(labels, predicted)

            # mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

            plt.axes(axes[i, j])
            plot_confusion_matrix(mat, classes=['B', 'E', 'F'], normalize=True,
                                  title='C={}, Clusters={}'.format(c, n_clusters))
    plt.tight_layout()

    pylab.savefig(path.join(project_folder, 'confusion_matrix.png'), dpi=fig.dpi)
    # plt.show()


if __name__ == '__main__':
    users = range(1, 34)

    MakeConfusionMatrix(users, [4, 6, 8, 10], 0, [0.1, 1, 10])

    # for clusters in [3, 4, 5, 6, 7, 8]:
    #     EvaluateRegression(users, clusters, 0)
    #     EvaluateRegressionPerCandidate(users, clusters, 0)
