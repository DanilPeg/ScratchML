import numpy as np
from ..base import BaseModel

class SGDSVMClassifier(BaseModel):
    def __init__(self, optimizer, regularizer=None, n_iter=1000, batch_size=32, fit_intercept=True, random_state=None, multi_class='ovr', C=1.0):
        self.optimizer = optimizer
        self.regularizer = regularizer
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.multi_class = multi_class
        self.C = C
        self.coef_ = None
        self.intercept_ = None
        self.classes_ = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        if n_classes == 2:
            y_bin = np.where(y == self.classes_[1], 1, -1)
            self._fit_binary(X, y_bin)
        else:
            # OvR
            self.coef_ = np.zeros((n_classes, X.shape[1]))
            self.intercept_ = np.zeros(n_classes)
            for i, cls in enumerate(self.classes_):
                y_bin = np.where(y == cls, 1, -1)
                coef, intercept = self._fit_binary(X, y_bin, return_coef=True)
                self.coef_[i] = coef
                self.intercept_[i] = intercept
        return self

    def _fit_binary(self, X, y, return_coef=False):
        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.random_state)
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
                margin = yb * (Xb @ w)
                mask = margin < 1
                grad = np.zeros_like(w)
                if np.any(mask):
                    grad += -self.C * np.mean((yb[mask, None] * Xb[mask]), axis=0)
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

    def decision_function(self, X):
        X = np.asarray(X)
        if self.coef_.ndim == 1:
            return X @ self.coef_ + self.intercept_
        else:
            return X @ self.coef_.T + self.intercept_

    def predict(self, X):
        scores = self.decision_function(X)
        if self.coef_.ndim == 1:
            return (np.sign(scores) + 1) // 2
        else:
            return self.classes_[np.argmax(scores, axis=1)]

class SGDSVMRegressor(BaseModel):
    def __init__(self, optimizer, regularizer=None, n_iter=1000, batch_size=32, fit_intercept=True, random_state=None, epsilon=0.1, C=1.0):
        self.optimizer = optimizer
        self.regularizer = regularizer
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.epsilon = epsilon
        self.C = C
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.random_state)
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
                margin = yb - Xb @ w
                mask = np.abs(margin) > self.epsilon
                grad = np.zeros_like(w)
                if np.any(mask):
                    grad += -self.C * np.mean(np.sign(margin[mask])[:, None] * Xb[mask], axis=0)
                if self.regularizer is not None:
                    grad[1:] += self.regularizer.grad(w[1:]) if self.fit_intercept else self.regularizer.grad(w)
                w = self.optimizer.update(w, grad)
        if self.fit_intercept:
            self.intercept_ = w[0]
            self.coef_ = w[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = w
        return self

    def predict(self, X):
        X = np.asarray(X)
        return X @ self.coef_ + self.intercept_ 