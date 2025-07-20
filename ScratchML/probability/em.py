import numpy as np
from ..base import BaseModel

class GaussianMixtureEM(BaseModel):
    def __init__(self, n_components=2, max_iter=100, tol=1e-4, reg_covar=1e-6, random_state=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state
        self.rng = np.random.RandomState(self.random_state)

    def _init_params(self, X):
        n_samples, n_features = X.shape
        # Инициализация центров через kmeans++
        centers = [X[self.rng.randint(n_samples)]]
        for _ in range(1, self.n_components):
            dists = np.min([np.sum((X - c) ** 2, axis=1) for c in centers], axis=0)
            probs = dists / dists.sum()
            idx = self.rng.choice(n_samples, p=probs)
            centers.append(X[idx])
        self.means_ = np.array(centers)
        self.covariances_ = np.array([np.cov(X.T) + self.reg_covar * np.eye(n_features) for _ in range(self.n_components)])
        self.weights_ = np.ones(self.n_components) / self.n_components

    def _gaussian(self, X, mean, cov):
        n_features = X.shape[1]
        det = np.linalg.det(cov)
        inv = np.linalg.inv(cov)
        norm = 1.0 / np.sqrt((2 * np.pi) ** n_features * det)
        diff = X - mean
        exp = np.exp(-0.5 * np.sum(diff @ inv * diff, axis=1))
        return norm * exp

    def fit(self, X):
        X = np.asarray(X)
        n_samples, n_features = X.shape
        self._init_params(X)
        log_likelihood = None
        for _ in range(self.max_iter):
            # E-step
            resp = np.zeros((n_samples, self.n_components))
            for k in range(self.n_components):
                resp[:, k] = self.weights_[k] * self._gaussian(X, self.means_[k], self.covariances_[k])
            resp_sum = resp.sum(axis=1, keepdims=True) + 1e-12
            resp /= resp_sum
            # M-step
            Nk = resp.sum(axis=0)
            self.weights_ = Nk / n_samples
            self.means_ = (resp.T @ X) / Nk[:, None]
            for k in range(self.n_components):
                diff = X - self.means_[k]
                self.covariances_[k] = (resp[:, k][:, None] * diff).T @ diff / Nk[k]
                self.covariances_[k] += self.reg_covar * np.eye(n_features)
            # Log-likelihood
            new_log_likelihood = np.sum(np.log(resp_sum))
            if log_likelihood is not None and np.abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        n_samples = X.shape[0]
        resp = np.zeros((n_samples, self.n_components))
        for k in range(self.n_components):
            resp[:, k] = self.weights_[k] * self._gaussian(X, self.means_[k], self.covariances_[k])
        resp_sum = resp.sum(axis=1, keepdims=True) + 1e-12
        resp /= resp_sum
        return resp

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1) 