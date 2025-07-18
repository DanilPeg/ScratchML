import numpy as np
from ..base import BaseOptimizer

class Momentum(BaseOptimizer):
    def __init__(self, lr: float = 0.01, momentum: float = 0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        if self.v is None:
            self.v = np.zeros_like(params)
        self.v = self.momentum * self.v - self.lr * grads
        return params + self.v 