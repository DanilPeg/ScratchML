import numpy as np
from ..base import BaseRegularizer

class L1(BaseRegularizer):
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def penalty(self, weights: np.ndarray) -> float:
        return self.alpha * np.sum(np.abs(weights))

    def grad(self, weights: np.ndarray) -> np.ndarray:
        return self.alpha * np.sign(weights) 