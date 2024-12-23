import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import plotly.express as px
from sklearn.datasets import make_classification

class PCA:
  def __init__(self, n_dimention: int):
    self.n_dimention = n_dimention

  def fit_transform(self, X):
    mean = np.mean(X, axis=0)
    X = X - mean
    cov = X.T.dot(X) / X.shape[0] 
    eigen_values, eigen_vectors, = np.linalg.eig(cov)
    select_index = np.argsort(eigen_values)[::-1][:self.n_dimention]
    U = eigen_vectors[:, select_index]
    X_new = X.dot(U)
    return X_new
    
if __name__ == "__main__":
  df = pd.read_csv(r"/home/greystone/StorageWall/data_debug/augment_data.csv")
  df = df.to_numpy()
  X = df[:,2:8]
  Y = df[:,8]

  pca = PCA(n_dimention=4)
  X = pca.fit_transform(X)


  #  for label in set(Y):
  #     X_class = new_X[Y == label]
  #     plt.scatter(X_class[:, 0], X_class[:, 1], label=label)

  y = Y

  # X, y = make_classification(
  #   n_features=12,
  #   n_classes=3,
  #   n_samples=1500,
  #   n_informative=2,
  #   random_state=5,
  #   n_clusters_per_class=1,
  # )

  fig = px.scatter_3d(x=X[:, 0], y=X[:, 1], z=X[:, 2], color=y, opacity=0.8)
  fig.show()

  plt.savefig('/home/greystone/StorageWall/data_debug/PCA.png')