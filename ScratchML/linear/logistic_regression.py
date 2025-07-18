import numpy as np
from ..base import BaseModel

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

class LogisticRegression(BaseModel):
    def __init__(self, optimizer, regularizer=None, n_iter=1000, batch_size=32, fit_intercept=True, random_state=None, multi_class='ovr'):
        self.optimizer = optimizer
        self.regularizer = regularizer
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.multi_class = multi_class  # 'ovr' (one-vs-rest - один против всех) или 'ovo' (все против всех)
        self.coef_ = None
        self.intercept_ = None
        self.classes_ = None
        self.models_ = None  # для OvO

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.random_state)
        if n_classes == 2:
            # Бинарная классификация
            y_bin = (y == self.classes_[1]).astype(float)
            self._fit_binary(X, y_bin, rng)
        elif self.multi_class == 'ovr':
            # One-vs-Rest
            self.coef_ = np.zeros((n_classes, n_features))
            self.intercept_ = np.zeros(n_classes)
            for i, cls in enumerate(self.classes_):
                y_bin = (y == cls).astype(float)
                coef, intercept = self._fit_binary(X, y_bin, rng, return_coef=True)
                self.coef_[i] = coef
                self.intercept_[i] = intercept
        elif self.multi_class == 'ovo':
            # One-vs-One
            from itertools import combinations
            self.models_ = {}
            for (i, cls_i), (j, cls_j) in combinations(enumerate(self.classes_), 2):
                idx = np.where((y == cls_i) | (y == cls_j))[0]
                X_pair = X[idx]
                y_pair = (y[idx] == cls_j).astype(float)
                model = LogisticRegression(
                    optimizer=self.optimizer,
                    regularizer=self.regularizer,
                    n_iter=self.n_iter,
                    batch_size=self.batch_size,
                    fit_intercept=self.fit_intercept,
                    random_state=self.random_state,
                    multi_class='ovr')
                model.fit(X_pair, y_pair)
                self.models_[(cls_i, cls_j)] = model
        else:
            raise ValueError(f"Unknown multi_class: {self.multi_class}")
        return self

    def _fit_binary(self, X, y, rng, return_coef=False):
        n_samples, n_features = X.shape
        if self.fit_intercept:
            X_ = np.hstack([np.ones((n_samples, 1)), X])
        else:
            X_ = X
        w = rng.randn(X_.shape[1]) * 0.01
        for epoch in range(self.n_iter):
            indices = rng.permutation(n_samples)
            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                batch_idx = indices[start:end]
                Xb = X_[batch_idx]
                yb = y[batch_idx]
                z = Xb @ w
                p = sigmoid(z)
                grad = Xb.T @ (p - yb) / len(yb)
                if self.regularizer is not None:
                    grad[1:] += self.regularizer.grad(w[1:]) if self.fit_intercept else self.regularizer.grad(w)
                w = self.optimizer.update(w, grad)
        if self.fit_intercept:
            intercept = w[0]
            coef = w[1:]
        else:
            intercept = 0.0
            coef = w
        if return_coef:
            return coef, intercept
        self.coef_ = coef
        self.intercept_ = intercept

    def predict_proba(self, X):
        X = np.asarray(X)
        n_classes = len(self.classes_)
        if n_classes == 2:
            z = X @ self.coef_ + self.intercept_ if self.fit_intercept else X @ self.coef_
            proba_1 = sigmoid(z)
            proba_0 = 1 - proba_1
            return np.vstack([proba_0, proba_1]).T
        elif self.multi_class == 'ovr':
            logits = X @ self.coef_.T + self.intercept_ if self.fit_intercept else X @ self.coef_.T
            proba = sigmoid(logits)
            proba = proba / proba.sum(axis=1, keepdims=True)
            return proba
        elif self.multi_class == 'ovo':
            # Для OvO: голосование
            votes = np.zeros((X.shape[0], len(self.classes_)))
            from itertools import combinations
            for (i, cls_i), (j, cls_j) in combinations(enumerate(self.classes_), 2):
                model = self.models_[(cls_i, cls_j)]
                pred = model.predict(X)
                votes[:, i] += (pred == 0)
                votes[:, j] += (pred == 1)
            proba = votes / votes.sum(axis=1, keepdims=True)
            return proba
        else:
            raise ValueError(f"Unknown multi_class: {self.multi_class}")

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)] 