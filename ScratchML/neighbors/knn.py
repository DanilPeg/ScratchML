import numpy as np
from ..base import BaseModel

# Евклидова метрика по умолчанию

def euclidean(a, b):
    return np.sqrt(np.sum((a - b) ** 2, axis=-1))

# Гауссово ядро

def gaussian_kernel(dist, sigma=1.0):
    return np.exp(-0.5 * (dist / sigma) ** 2)

class KNNRegressor(BaseModel):
    def __init__(self, n_neighbors=5, metric=euclidean, kernel=None, kernel_params=None):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.kernel = kernel  # None (обычный), либо функция ядра
        self.kernel_params = kernel_params or {}

    def fit(self, X, y):
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
        return self

    def predict(self, X):
        X = np.asarray(X)
        preds = []
        for x in X:
            dists = self.metric(self.X_train, x)
            idx = np.argsort(dists)[:self.n_neighbors]
            if self.kernel is None:
                weights = np.ones_like(idx, dtype=float)
            else:
                weights = self.kernel(dists[idx], **self.kernel_params)
            if np.sum(weights) == 0:
                pred = np.mean(self.y_train[idx])
            else:
                pred = np.sum(self.y_train[idx] * weights) / np.sum(weights)
            preds.append(pred)
        return np.array(preds)

class KNNClassifier(BaseModel):
    def __init__(self, n_neighbors=5, metric=euclidean, kernel=None, kernel_params=None):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.kernel = kernel  # None (обычный), либо функция ядра
        self.kernel_params = kernel_params or {}

    def fit(self, X, y):
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
        self.classes_ = np.unique(self.y_train)
        return self

    def predict(self, X):
        X = np.asarray(X)
        preds = []
        for x in X:
            dists = self.metric(self.X_train, x)
            idx = np.argsort(dists)[:self.n_neighbors]
            if self.kernel is None:
                weights = np.ones_like(idx, dtype=float)
            else:
                weights = self.kernel(dists[idx], **self.kernel_params)
            votes = {}
            for c in self.classes_:
                mask = (self.y_train[idx] == c)
                votes[c] = np.sum(weights[mask])
            pred = max(votes, key=votes.get)
            preds.append(pred)
        return np.array(preds) 