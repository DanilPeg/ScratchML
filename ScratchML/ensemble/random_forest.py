import numpy as np
from ..tree.cart import DecisionTreeClassifier, DecisionTreeRegressor

class BaseRandomForest:
    def __init__(self, n_estimators=100, max_features='sqrt', bootstrap=True, random_state=None, tree_params=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.tree_params = tree_params or {}
        self.trees = []
        self.features_ = []
        self.rng = np.random.RandomState(self.random_state)

    def _get_max_features(self, n_features):
        if self.max_features == 'sqrt':
            return max(1, int(np.sqrt(n_features)))
        elif self.max_features == 'log2':
            return max(1, int(np.log2(n_features)))
        elif isinstance(self.max_features, int):
            return min(n_features, self.max_features)
        elif isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_features))
        else:
            return n_features

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples, n_features = X.shape
        self.trees = []
        self.features_ = []
        for i in range(self.n_estimators):
            if self.bootstrap:
                idx = self.rng.choice(n_samples, n_samples, replace=True)
            else:
                idx = np.arange(n_samples)
            X_sample = X[idx]
            y_sample = y[idx]
            max_feats = self._get_max_features(n_features)
            feats = self.rng.choice(n_features, max_feats, replace=False)
            self.features_.append(feats)
            tree = self._make_tree()
            tree.fit(X_sample[:, feats], y_sample)
            self.trees.append(tree)
        return self

    def _predict_trees(self, X):
        preds = []
        for tree, feats in zip(self.trees, self.features_):
            preds.append(tree.predict(X[:, feats]))
        return np.array(preds)

class RandomForestClassifier(BaseRandomForest):
    def __init__(self, n_estimators=100, max_features='sqrt', bootstrap=True, random_state=None, tree_params=None):
        super().__init__(n_estimators, max_features, bootstrap, random_state, tree_params)
    def _make_tree(self):
        return DecisionTreeClassifier(**self.tree_params)
    def predict(self, X):
        tree_preds = self._predict_trees(X)
        from scipy.stats import mode
        maj = mode(tree_preds, axis=0, keepdims=False).mode
        return maj

class RandomForestRegressor(BaseRandomForest):
    def __init__(self, n_estimators=100, max_features='sqrt', bootstrap=True, random_state=None, tree_params=None):
        super().__init__(n_estimators, max_features, bootstrap, random_state, tree_params)
    def _make_tree(self):
        return DecisionTreeRegressor(**self.tree_params)
    def predict(self, X):
        tree_preds = self._predict_trees(X)
        return np.mean(tree_preds, axis=0) 