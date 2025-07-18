import numpy as np

class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, X):
        X = np.asarray(X)
        n_samples = X.shape[0]
        labels = np.full(n_samples, -1, dtype=int)  # -1 — шум
        cluster_id = 0
        visited = np.zeros(n_samples, dtype=bool)
        for i in range(n_samples):
            if visited[i]:
                continue
            visited[i] = True
            neighbors = self._region_query(X, i)
            if len(neighbors) < self.min_samples:
                labels[i] = -1  # шум
            else:
                self._expand_cluster(X, labels, i, neighbors, cluster_id, visited)
                cluster_id += 1
        self.labels_ = labels
        self.n_clusters_ = cluster_id
        return self

    def _region_query(self, X, idx):
        dists = np.linalg.norm(X - X[idx], axis=1)
        return np.where(dists <= self.eps)[0]

    def _expand_cluster(self, X, labels, idx, neighbors, cluster_id, visited):
        labels[idx] = cluster_id
        i = 0
        while i < len(neighbors):
            n_idx = neighbors[i]
            if not visited[n_idx]:
                visited[n_idx] = True
                n_neighbors = self._region_query(X, n_idx)
                if len(n_neighbors) >= self.min_samples:
                    neighbors = np.concatenate([neighbors, n_neighbors[~np.isin(n_neighbors, neighbors)]])
            if labels[n_idx] == -1:
                labels[n_idx] = cluster_id
            i += 1 