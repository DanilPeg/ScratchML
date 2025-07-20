import numpy as np
from scipy.linalg import lstsq
from ..base import BaseModel

class LinearRegression(BaseModel):
    """Линейная регрессия с выбором solver: 'lstsq' (по умолчанию, быстрый и устойчивый) или 'svd' (ручной SVD)."""
    def __init__(self, fit_intercept: bool = True, solver: str = 'lstsq'):
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X)
        y = np.asarray(y)
        if self.fit_intercept:
            X_ = np.hstack([np.ones((X.shape[0], 1)), X])
        else:
            X_ = X
        if self.solver == 'lstsq':
            w, *_ = lstsq(X_, y)
        elif self.solver == 'svd':
            U, s, Vt = np.linalg.svd(X_, full_matrices=False)
            s_inv = np.diag(1 / s)
            X_pinv = Vt.T @ s_inv @ U.T
            w = X_pinv @ y
        else:
            raise ValueError(f"Unknown solver: {self.solver}")
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