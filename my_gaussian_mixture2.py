import numpy as np
from sklearn.base import BaseEstimator
from sklearn.mixture import GaussianMixture


def get_initial_means(X, n_components, init_params="kmeans", r=0):
    # Run a GaussianMixture with max_iter=0 to output the initialization means
    gmm = GaussianMixture(
        n_components=n_components, init_params=init_params, tol=1e-9, max_iter=0, random_state=r
    ).fit(X)
    return gmm.means_

def map_to_class(id_components, n_classes=2):
    return int(id_components / n_classes)

class MyGaussianMixture2(BaseEstimator):
  
  def __init__(self, n_components=1):
    self.n_components = n_components

  def fit(self, X, y):
    # find number of classes
    self.n_classes = int(y.max() + 1)
    
    self.gm_means = [None] * self.n_classes
    means_init = []
    # calculate means for each class
    for c in range(self.n_classes):
      # find the correspond items
      X_c = X[np.where(y == c)]
      # calculate means for each classes
      means_c = get_initial_means(X_c, self.n_components)
      self.gm_means[c] = means_c
      means_init.extend(means_c)
    
    # print('means_init', means_init)
    self.gmm = GaussianMixture(n_components=self.n_classes * self.n_components, means_init=means_init, random_state=0)
    
    self.gmm.fit(X)

  def predict0(self, X):
    y_pre = self.gmm.predict(X)
    y_pre_n = [None] * len(y_pre)
    
    for i in range(0,len(y_pre)):
      y_pre_n[i] = int(y_pre[i] / self.n_classes)
    
    return y_pre_n

  def predict_proba(self, X):
    y_pre = self.gmm.predict_proba(X)
    y_pre_n = [None] * len(y_pre)
    
    for i, ye in enumerate(y_pre):
      ye_arr = np.array_split(ye, self.n_components)
      new_arr = []
      
      for _, arr in enumerate(ye_arr):
        new_arr.append(np.sum(arr))

      y_pre_n[i] = new_arr
    
    return y_pre_n
  
  def predict(self, X):
    y_pre_prob = self.predict_proba(X)

    y_pre_n = np.array(y_pre_prob).argmax(axis=1)
    return y_pre_n