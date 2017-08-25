import sklearn
import numpy as np
from cyvlfeat.fisher import fisher


# train_data: array-like, shape (n_samples, n_features)
def get_gmm(train_data):
    return sklearn.mixture.GMM(n_components=3, covariance_type='diag', verbose=1).fit(train_data)


def fv_encoding(gmm, sample):
    means = np.float32(np.transpose(gmm.means_))
    covars = np.float32(np.transpose(gmm.covars_))
    priors = np.float32(np.transpose(gmm.weights_))
    sample = np.float32(sample)
    encoded_sample = fisher(sample, means, covars, priors, verbose=False, improved=True)
    return encoded_sample
