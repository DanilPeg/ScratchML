import numpy as np
from ..base import BaseModel

class PCA(BaseModel):
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.mean_ = None

    def fit(self, X):
        X = np.asarray(X)
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        if self.n_components is None:
            n_comp = X.shape[1]
        else:
            n_comp = self.n_components
        self.components_ = Vt[:n_comp]
        self.explained_variance_ = (S[:n_comp] ** 2) / (X.shape[0] - 1)
        return self

    def transform(self, X):
        X = np.asarray(X)
        Xc = X - self.mean_
        return Xc @ self.components_.T

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X) 