import numpy as np
from sklearn.base import BaseEstimator
from sklearn.mixture import GaussianMixture


def get_initial_means(X, n_components, init_params="kmeans", r=0):
    # Run a GaussianMixture with max_iter=0 to output the initialization means
    gmm = GaussianMixture(
        n_components=n_components, init_params=init_params, tol=1e-9, max_iter=0, random_state=r
    ).fit(X)
    return gmm.means_

def map_to_class(id_components, n_classes):
    return int(id_components / n_classes)

class MyGaussianMixture2(BaseEstimator):
  
  def __init__(self, n_components=1):
    self.n_components = n_components

  def fit(self, X, y):
    # find number of classes
    self.n_classes = int(y.max() + 1)
    
    self.gm_means = []
    means_init = []
    # calculate means for each class
    for c in range(self.n_classes):
      # find the correspond items
      X_c = X[np.where(y == c)]
      # calculate means for each classes
      means_c = get_initial_means(X_c, self.n_components)
      self.gm_means[c] = means_c
      means_init.append(means_c)
    
    self.gmm = GaussianMixture(n_components=self.n_classes * self.n_components, means_init=means_init)
    
    self.gmm.fit(X)

  def predict(self, X):
    y_pre = self.predict(X)
    map_to_class_func = np.vectorize(map_to_class)
    return map_to_class_func(y_pre)