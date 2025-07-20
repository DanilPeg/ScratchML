import numpy as np
from ..base import BaseModel

class GaussianNB(BaseModel):
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        self.theta_ = np.zeros((n_classes, n_features))  # средние
        self.sigma_ = np.zeros((n_classes, n_features))  # дисперсии
        self.class_prior_ = np.zeros(n_classes)
        for idx, c in enumerate(self.classes_):
            Xc = X[y == c]
            self.theta_[idx] = Xc.mean(axis=0)
            self.sigma_[idx] = Xc.var(axis=0) + 1e-9  # для устойчивости
            self.class_prior_[idx] = Xc.shape[0] / X.shape[0]
        return self

    def _joint_log_likelihood(self, X):
        X = np.asarray(X)
        jll = []
        for idx in range(len(self.classes_)):
            prior = np.log(self.class_prior_[idx])
            nll = -0.5 * np.sum(np.log(2 * np.pi * self.sigma_[idx]))
            nll -= 0.5 * np.sum(((X - self.theta_[idx]) ** 2) / self.sigma_[idx], axis=1)
            jll.append(prior + nll)
        return np.array(jll).T  # shape (n_samples, n_classes)

    def predict(self, X):
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]

    def predict_proba(self, X):
        jll = self._joint_log_likelihood(X)
        log_prob = jll - jll.max(axis=1, keepdims=True)
        prob = np.exp(log_prob)
        prob /= prob.sum(axis=1, keepdims=True)
        return prob 