import numpy as np
from ..base import BaseModel

class LinearRegression(BaseModel):
    """Линейная регрессия с аналитическим решением через SVD."""
    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X)
        y = np.asarray(y)
        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        s_inv = np.diag(1 / s)
        X_pinv = Vt.T @ s_inv @ U.T
        w = X_pinv @ y
        if self.fit_intercept:
            self.intercept_ = w[0]
            self.coef_ = w[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = w
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if self.fit_intercept:
            return X @ self.coef_ + self.intercept_
        else:
            return X @ self.coef_ 