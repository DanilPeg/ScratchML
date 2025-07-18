import numpy as np

class AgglomerativeClustering:
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters

    def fit(self, X):
        X = np.asarray(X)
        n_samples = X.shape[0]
        clusters = [{i} for i in range(n_samples)]
        dists = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        labels = np.arange(n_samples)
        while len(clusters) > self.n_clusters:
            min_dist = np.inf
            to_merge = (None, None)
            for i in range(len(clusters)):
                for j in range(i+1, len(clusters)):
                    dist = np.min(dists[np.ix_(list(clusters[i]), list(clusters[j]))])
                    if dist < min_dist:
                        min_dist = dist
                        to_merge = (i, j)
            i, j = to_merge
            clusters[i] = clusters[i].union(clusters[j])
            del clusters[j]
        labels_out = np.empty(n_samples, dtype=int)
        for idx, cluster in enumerate(clusters):
            for i in cluster:
                labels_out[i] = idx
        self.labels_ = labels_out
        return self 