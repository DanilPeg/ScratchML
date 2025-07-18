import numpy as np
from ..base import BaseOptimizer

class Nesterov(BaseOptimizer):
    def __init__(self, lr: float = 0.01, momentum: float = 0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        if self.v is None:
            self.v = np.zeros_like(params)
        v_prev = self.v.copy()
        self.v = self.momentum * self.v - self.lr * grads
        return params + (-self.momentum * v_prev + (1 + self.momentum) * self.v) 