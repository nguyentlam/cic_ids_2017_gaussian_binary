import numpy as np
from sklearn.base import BaseEstimator
from sklearn.mixture import GaussianMixture


class MyGaussianMixture(BaseEstimator):
  
  def __init__(self, n_components=1):
    self.n_components = n_components

  def fit(self, X, y):
    # find number of classes
    self.n_classes = int(y.max() + 1)
    
    # create a GM for each class
    self.gm_densities = [GaussianMixture(self.n_components, covariance_type='full') for _ in range(self.n_classes)]
    
    # fit the Mixture densities for each class
    for c in range(self.n_classes):
      # find the correspond items
      temp = X[np.where(y == c)]
      # estimate density parameters using EM
      self.gm_densities[c].fit(temp)

  def predict(self, X):
    # calculate log likelihood for each class
    log_likelihoods = np.hstack([ self.gm_densities[c].score_samples(X).reshape((-1, 1)) for c in range(self.n_classes) ])
    print('log_likelihoods', log_likelihoods)
    # return the class whose density maximizes the log likelihoods
    class_idx = log_likelihoods.argmax(axis=1)
    return class_idx