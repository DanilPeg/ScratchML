import numpy as np
from ..tree.cart import DecisionTreeRegressor

class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, subsample=1.0, random_state=None, tree_params=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.random_state = random_state
        self.tree_params = tree_params or {}
        self.trees = []
        self.rng = np.random.RandomState(self.random_state)

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples = X.shape[0]
        self.init_ = np.mean(y)
        self.trees = []
        F = np.full(n_samples, self.init_)
        for m in range(self.n_estimators):
            if self.subsample < 1.0:
                idx = self.rng.choice(n_samples, int(n_samples * self.subsample), replace=False)
            else:
                idx = np.arange(n_samples)
            residual = y[idx] - F[idx]
            tree = DecisionTreeRegressor(max_depth=self.max_depth, **self.tree_params)
            tree.fit(X[idx], residual)
            self.trees.append(tree)
            F += self.learning_rate * tree.predict(X)
        return self

    def predict(self, X):
        X = np.asarray(X)
        pred = np.full(X.shape[0], self.init_)
        for tree in self.trees:
            pred += self.learning_rate * tree.predict(X)
        return pred

class GradientBoostingClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, subsample=1.0, random_state=None, tree_params=None, multi_class='ovr'):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.random_state = random_state
        self.tree_params = tree_params or {}
        self.trees = []
        self.rng = np.random.RandomState(self.random_state)
        self.multi_class = multi_class

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_samples = X.shape[0]
        if n_classes == 2 and self.multi_class != 'softmax':
            y_bin = (y == self.classes_[1]).astype(float)
            self._fit_binary(X, y_bin)
        elif self.multi_class == 'softmax':
            # Делаем мультикласс через софтмакс
            Y = np.zeros((n_samples, n_classes))
            for i, c in enumerate(self.classes_):
                Y[:, i] = (y == c).astype(float)
            self.init_ = np.log(np.mean(Y, axis=0) + 1e-12)
            self.trees = [[] for _ in range(n_classes)]
            F = np.tile(self.init_, (n_samples, 1))
            for m in range(self.n_estimators):
                P = np.exp(F - F.max(axis=1, keepdims=True))
                P = P / P.sum(axis=1, keepdims=True)
                for k in range(n_classes):
                    if self.subsample < 1.0:
                        idx = self.rng.choice(n_samples, int(n_samples * self.subsample), replace=False)
                    else:
                        idx = np.arange(n_samples)
                    grad = Y[idx, k] - P[idx, k]
                    tree = DecisionTreeRegressor(max_depth=self.max_depth, **self.tree_params)
                    tree.fit(X[idx], grad)
                    self.trees[k].append(tree)
                    F[:, k] += self.learning_rate * tree.predict(X)
        else:
            # Один против всех
            self.ovr_models_ = []
            for c in self.classes_:
                y_bin = (y == c).astype(float)
                model = GradientBoostingClassifier(
                    n_estimators=self.n_estimators,
                    learning_rate=self.learning_rate,
                    max_depth=self.max_depth,
                    subsample=self.subsample,
                    random_state=self.random_state,
                    tree_params=self.tree_params)
                model._fit_binary(X, y_bin)
                self.ovr_models_.append(model)
        return self

    def _fit_binary(self, X, y):
        n_samples = X.shape[0]
        self.init_ = np.log(np.mean(y) / (1 - np.mean(y) + 1e-12))
        self.trees = []
        F = np.full(n_samples, self.init_)
        for m in range(self.n_estimators):
            if self.subsample < 1.0:
                idx = self.rng.choice(n_samples, int(n_samples * self.subsample), replace=False)
            else:
                idx = np.arange(n_samples)
            p = 1 / (1 + np.exp(-F[idx]))
            grad = y[idx] - p
            tree = DecisionTreeRegressor(max_depth=self.max_depth, **self.tree_params)
            tree.fit(X[idx], grad)
            self.trees.append(tree)
            F += self.learning_rate * tree.predict(X)

    def predict_proba(self, X):
        X = np.asarray(X)
        n_classes = len(self.classes_)
        if n_classes == 2 and not hasattr(self, 'ovr_models_') and self.multi_class != 'softmax':
            F = np.full(X.shape[0], self.init_)
            for tree in self.trees:
                F += self.learning_rate * tree.predict(X)
            proba_1 = 1 / (1 + np.exp(-F))
            proba_0 = 1 - proba_1
            return np.vstack([proba_0, proba_1]).T
        elif self.multi_class == 'softmax':
            F = np.tile(self.init_, (X.shape[0], 1))
            for k, trees in enumerate(self.trees):
                for tree in trees:
                    F[:, k] += self.learning_rate * tree.predict(X)
            P = np.exp(F - F.max(axis=1, keepdims=True))
            P = P / P.sum(axis=1, keepdims=True)
            return P
        else:
            probas = np.column_stack([m.predict_proba(X)[:, 1] for m in self.ovr_models_])
            probas = probas / probas.sum(axis=1, keepdims=True)
            return probas

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)] 