import numpy as np
from ..base import BaseRegularizer

class ElasticNet(BaseRegularizer):
    def __init__(self, alpha=1.0, l1_ratio=0.5):
        self.alpha = alpha
        self.l1_ratio = l1_ratio  # 0 — только L2, 1 — только L1

    def penalty(self, weights: np.ndarray) -> float:
        l1 = np.sum(np.abs(weights))
        l2 = 0.5 * np.sum(weights ** 2)
        return self.alpha * (self.l1_ratio * l1 + (1 - self.l1_ratio) * l2)

    def grad(self, weights: np.ndarray) -> np.ndarray:
        l1_grad = np.sign(weights)
        l2_grad = weights
        return self.alpha * (self.l1_ratio * l1_grad + (1 - self.l1_ratio) * l2_grad) 