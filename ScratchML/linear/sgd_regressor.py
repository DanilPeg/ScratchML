import numpy as np
from ..base import BaseModel

class SGDRegressor(BaseModel):
    def __init__(self, optimizer, regularizer=None, n_iter=1000, batch_size=32, fit_intercept=True, random_state=None):
        self.optimizer = optimizer
        self.regularizer = regularizer
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.fit_intercept = fit_intercept
        self.random_state = random_state
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
                y_pred = Xb @ w
                grad = Xb.T @ (y_pred - yb) / len(yb)
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
        if self.fit_intercept:
            return X @ self.coef_ + self.intercept_
        else:
            return X @ self.coef_ 