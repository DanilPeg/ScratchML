import numpy as np
from .kernels import kernel_linear, kernel_poly, kernel_rbf

class SVC:
    def __init__(self, kernel='linear', C=1.0, max_iter=1000, degree=3, gamma=1.0, random_state=None):
        self.kernel_name = kernel
        self.C = C
        self.max_iter = max_iter
        self.degree = degree
        self.gamma = gamma
        self.random_state = random_state
        self.rng = np.random.RandomState(self.random_state)
        self._kernel = self._get_kernel()

    def _get_kernel(self):
        if self.kernel_name == 'linear':
            return kernel_linear
        elif self.kernel_name == 'poly':
            return lambda x, y: kernel_poly(x, y, degree=self.degree)
        elif self.kernel_name == 'rbf':
            return lambda x, y: kernel_rbf(x, y, gamma=self.gamma)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel_name}")

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
            self.ovr_models_ = []
            for cls in self.classes_:
                y_bin = np.where(y == cls, 1, -1)
                model = SVC(kernel=self.kernel_name, C=self.C, max_iter=self.max_iter, degree=self.degree, gamma=self.gamma, random_state=self.random_state)
                model._fit_binary(X, y_bin)
                self.ovr_models_.append(model)
        return self

    def _fit_binary(self, X, y):
        n_samples = X.shape[0]
        lambdas = np.zeros(n_samples)
        K = self._kernel(X, X) * y[:, None] * y[None, :]
        for _ in range(self.max_iter):
            for idxM in range(n_samples):
                idxL = self.rng.randint(0, n_samples)
                Q = K[[[idxM, idxM], [idxL, idxL]], [[idxM, idxL], [idxM, idxL]]]
                v0 = lambdas[[idxM, idxL]]
                k0 = 1 - np.sum(lambdas * K[[idxM, idxL]], axis=1)
                u = np.array([-y[idxL], y[idxM]])
                t_max = np.dot(k0, u) / (np.dot(np.dot(Q, u), u) + 1E-15)
                t = self._restrict_to_square(t_max, v0, u)
                lambdas[[idxM, idxL]] = v0 + u * t
        idx = np.nonzero(lambdas > 1E-8)[0]
        self.lambdas_ = lambdas
        self.X_sv_ = X[idx]
        self.y_sv_ = y[idx]
        self.lambdas_sv_ = lambdas[idx]
        self.K_sv_ = self._kernel(self.X_sv_, self.X_sv_) * self.y_sv_[:, None] * self.y_sv_[None, :]
        self.b_ = np.mean((1.0 - np.sum(self.K_sv_ * self.lambdas_sv_, axis=1)) * self.y_sv_)

    def _restrict_to_square(self, t, v0, u):
        t = (np.clip(v0 + t*u, 0, self.C) - v0)[1]/u[1]
        return (np.clip(v0 + t*u, 0, self.C) - v0)[0]/u[0]

    def decision_function(self, X):
        X = np.asarray(X)
        if hasattr(self, 'ovr_models_'):
            return np.column_stack([m.decision_function(X) for m in self.ovr_models_])
        else:
            K = self._kernel(X, self.X_sv_)
            return np.sum(K * self.y_sv_ * self.lambdas_sv_, axis=1) + self.b_

    def predict(self, X):
        scores = self.decision_function(X)
        if hasattr(self, 'ovr_models_'):
            return self.classes_[np.argmax(scores, axis=1)]
        else:
            return (np.sign(scores) + 1) // 2 

class SVR:
    def __init__(self, kernel='linear', C=1.0, epsilon=0.1, max_iter=1000, degree=3, gamma=1.0, random_state=None):
        self.kernel_name = kernel
        self.C = C
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.degree = degree
        self.gamma = gamma
        self.random_state = random_state
        self.rng = np.random.RandomState(self.random_state)
        self._kernel = self._get_kernel()

    def _get_kernel(self):
        if self.kernel_name == 'linear':
            return kernel_linear
        elif self.kernel_name == 'poly':
            return lambda x, y: kernel_poly(x, y, degree=self.degree)
        elif self.kernel_name == 'rbf':
            return lambda x, y: kernel_rbf(x, y, gamma=self.gamma)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel_name}")

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples = X.shape[0]
        lambdas_p = np.zeros(n_samples)
        lambdas_n = np.zeros(n_samples)
        K = self._kernel(X, X)
        for _ in range(self.max_iter):
            for idxM in range(n_samples):
                idxL = self.rng.randint(0, n_samples)
                # SMO-подобное обновление для SVR
                Q = K[[[idxM, idxM], [idxL, idxL]], [[idxM, idxL], [idxM, idxL]]]
                v0 = np.array([lambdas_p[idxM] - lambdas_n[idxM], lambdas_p[idxL] - lambdas_n[idxL]])
                k0 = y[[idxM, idxL]] - np.sum((lambdas_p - lambdas_n) * K[[idxM, idxL]], axis=1) - self.epsilon * np.array([1, 1])
                u = np.array([1, -1])
                t_max = np.dot(k0, u) / (np.dot(np.dot(Q, u), u) + 1E-15)
                t = self._restrict_to_square(t_max, v0, u)
                lambdas_p[idxM] = np.clip(lambdas_p[idxM] + t, 0, self.C)
                lambdas_n[idxL] = np.clip(lambdas_n[idxL] - t, 0, self.C)
        self.lambdas_p_ = lambdas_p
        self.lambdas_n_ = lambdas_n
        self.X_sv_ = X
        self.y_sv_ = y
        # bias (b) вычисляем по support vectors
        mask = (np.abs(lambdas_p - lambdas_n) > 1E-8)
        if np.any(mask):
            self.b_ = np.mean(y[mask] - np.sum((lambdas_p - lambdas_n)[None, :] * K[mask], axis=1))
        else:
            self.b_ = 0.0
        return self

    def _restrict_to_square(self, t, v0, u):
        # Ограничение для SVR (аналогично SVC)
        t = (np.clip(v0 + t*u, -self.C, self.C) - v0)[1]/u[1]
        return (np.clip(v0 + t*u, -self.C, self.C) - v0)[0]/u[0]

    def predict(self, X):
        X = np.asarray(X)
        K = self._kernel(X, self.X_sv_)
        return np.sum(K * (self.lambdas_p_ - self.lambdas_n_), axis=1) + self.b_ 