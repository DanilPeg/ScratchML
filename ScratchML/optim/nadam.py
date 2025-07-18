import numpy as np
from ..base import BaseOptimizer

class Nadam(BaseOptimizer):
    def __init__(self, lr: float = 0.002, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0

    def update(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        if self.m is None:
            self.m = np.zeros_like(params)
        if self.v is None:
            self.v = np.zeros_like(params)
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        m_hat = self.m / (1 - self.beta1 ** self.t)
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        m_nesterov = self.beta1 * m_hat + (1 - self.beta1) * grads / (1 - self.beta1 ** self.t)
        return params - self.lr * m_nesterov / (np.sqrt(v_hat) + self.eps) 