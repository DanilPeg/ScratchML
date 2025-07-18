import numpy as np
from ..base import BaseOptimizer

class RMSProp(BaseOptimizer):
    def __init__(self, lr: float = 0.001, beta: float = 0.9, eps: float = 1e-8):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.s = None

    def update(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        if self.s is None:
            self.s = np.zeros_like(params)
        self.s = self.beta * self.s + (1 - self.beta) * grads ** 2
        return params - self.lr * grads / (np.sqrt(self.s) + self.eps) 