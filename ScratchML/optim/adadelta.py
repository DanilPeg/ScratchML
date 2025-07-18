import numpy as np
from ..base import BaseOptimizer

class AdaDelta(BaseOptimizer):
    def __init__(self, rho: float = 0.95, eps: float = 1e-6):
        self.rho = rho
        self.eps = eps
        self.Eg = None
        self.Edx = None

    def update(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        if self.Eg is None:
            self.Eg = np.zeros_like(params)
        if self.Edx is None:
            self.Edx = np.zeros_like(params)
        self.Eg = self.rho * self.Eg + (1 - self.rho) * grads ** 2
        dx = - (np.sqrt(self.Edx + self.eps) / np.sqrt(self.Eg + self.eps)) * grads
        self.Edx = self.rho * self.Edx + (1 - self.rho) * dx ** 2
        return params + dx 