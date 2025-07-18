import numpy as np
from ..base import BaseRegularizer

class L2(BaseRegularizer):
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def penalty(self, weights: np.ndarray) -> float:
        return 0.5 * self.alpha * np.sum(weights ** 2)

    def grad(self, weights: np.ndarray) -> np.ndarray:
        return self.alpha * weights 