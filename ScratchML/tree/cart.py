import numpy as np
from ..base import BaseModel

def gini(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return 1 - np.sum(p ** 2)

def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return -np.sum(p * np.log2(p + 1e-12))

def mse(y):
    return np.mean((y - np.mean(y)) ** 2)

def mae(y):
    return np.mean(np.abs(y - np.mean(y)))

class Node:
    def __init__(self, *, is_leaf, prediction=None, feature_index=None, threshold=None, left=None, right=None, n_samples=None, impurity=None):
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.n_samples = n_samples
        self.impurity = impurity

class BaseDecisionTree(BaseModel):
    def __init__(self, criterion, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_leaf_nodes=None, ccp_alpha=0.0):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.ccp_alpha = ccp_alpha
        self.root = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.n_classes_ = len(np.unique(y)) if self._is_classification else None
        self.n_features_ = X.shape[1]
        self.root = self._grow_tree(X, y, depth=0)
        if self.ccp_alpha > 0.0:
            self._prune(self.root)
        return self

    def predict(self, X):
        X = np.asarray(X)
        return np.array([self._predict_row(x, self.root) for x in X])

    def _predict_row(self, x, node):
        while not node.is_leaf:
            if x[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.prediction

    def _grow_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        impurity = self.criterion(y)
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           n_samples <= self.min_samples_leaf or \
           (self.max_leaf_nodes is not None and self._leaf_count >= self.max_leaf_nodes):
            return Node(is_leaf=True, prediction=self._leaf_value(y), n_samples=n_samples, impurity=impurity)
        best_feat, best_thr, best_gain = None, None, -np.inf
        for feature in range(n_features):
            thresholds, classes = zip(*sorted(zip(X[:, feature], y)))
            for i in range(self.min_samples_leaf, n_samples - self.min_samples_leaf + 1):
                if thresholds[i] == thresholds[i - 1]:
                    continue
                left_y = classes[:i]
                right_y = classes[i:]
                if len(left_y) < self.min_samples_leaf or len(right_y) < self.min_samples_leaf:
                    continue
                gain = impurity - (len(left_y) / n_samples) * self.criterion(left_y) - (len(right_y) / n_samples) * self.criterion(right_y)
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feature
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2
        if best_gain == -np.inf:
            return Node(is_leaf=True, prediction=self._leaf_value(y), n_samples=n_samples, impurity=impurity)
        idx_left = X[:, best_feat] < best_thr
        left = self._grow_tree(X[idx_left], y[idx_left], depth + 1)
        right = self._grow_tree(X[~idx_left], y[~idx_left], depth + 1)
        return Node(is_leaf=False, feature_index=best_feat, threshold=best_thr, left=left, right=right, n_samples=n_samples, impurity=impurity)

    def _prune(self, node):
        if node.is_leaf:
            return node, node.n_samples * node.impurity
        left, left_risk = self._prune(node.left)
        right, right_risk = self._prune(node.right)
        node.left = left
        node.right = right
        risk = left_risk + right_risk
        leaf_risk = node.n_samples * self.criterion([node.prediction] * node.n_samples)
        if (risk + self.ccp_alpha) >= leaf_risk:
            return Node(is_leaf=True, prediction=node.prediction, n_samples=node.n_samples, impurity=node.impurity), leaf_risk
        return node, risk

    @property
    def _leaf_count(self):
        def count(node):
            if node.is_leaf:
                return 1
            return count(node.left) + count(node.right)
        return count(self.root) if self.root else 0

class DecisionTreeClassifier(BaseDecisionTree):
    _is_classification = True
    def __init__(self, criterion="gini", **kwargs):
        crit = gini if criterion == "gini" else entropy
        super().__init__(crit, **kwargs)
    def _leaf_value(self, y):
        vals, counts = np.unique(y, return_counts=True)
        return vals[np.argmax(counts)]

class DecisionTreeRegressor(BaseDecisionTree):
    _is_classification = False
    def __init__(self, criterion="mse", **kwargs):
        crit = mse if criterion == "mse" else mae
        super().__init__(crit, **kwargs)
    def _leaf_value(self, y):
        return np.mean(y) 