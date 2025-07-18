import numpy as np
from ..base import BaseOptimizer

class SGD(BaseOptimizer):
    def __init__(self, lr: float = 0.01):
        self.lr = lr

    def update(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        return params - self.lr * grads 