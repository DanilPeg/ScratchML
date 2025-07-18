import numpy as np

class KMeans:
    def __init__(self, n_clusters=8, n_init=10, max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.rng = np.random.RandomState(self.random_state)

    def _init_centers_kmeanspp(self, X):
        n_samples = X.shape[0]
        centers = []
        # Первый центр случайно
        idx = self.rng.randint(n_samples)
        centers.append(X[idx])
        for _ in range(1, self.n_clusters):
            dists = np.min([np.sum((X - c) ** 2, axis=1) for c in centers], axis=0)
            probs = dists / dists.sum()
            idx = self.rng.choice(n_samples, p=probs)
            centers.append(X[idx])
        return np.array(centers)

    def fit(self, X):
        X = np.asarray(X)
        best_inertia = None
        best_centers = None
        best_labels = None
        for _ in range(self.n_init):
            centers = self._init_centers_kmeanspp(X)
            for i in range(self.max_iter):
                # Вычисляем расстояния и назначаем кластеры
                dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
                labels = np.argmin(dists, axis=1)
                new_centers = np.array([X[labels == k].mean(axis=0) if np.any(labels == k) else centers[k] for k in range(self.n_clusters)])
                shift = np.linalg.norm(new_centers - centers)
                centers = new_centers
                if shift < self.tol:
                    break
            inertia = np.sum((X - centers[labels]) ** 2)
            if best_inertia is None or inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers.copy()
                best_labels = labels.copy()
        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        return self

    def predict(self, X):
        X = np.asarray(X)
        dists = np.linalg.norm(X[:, None, :] - self.cluster_centers_[None, :, :], axis=2)
        return np.argmin(dists, axis=1) 