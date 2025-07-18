import numpy as np
from abc import ABC, abstractmethod

class BaseModel(ABC):
    """Базовый класс для всех моделей."""
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

class BaseOptimizer(ABC):
    """Базовый класс для оптимизаторов."""
    @abstractmethod
    def update(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        pass

class BaseRegularizer(ABC):
    """Базовый класс для регуляризаторов."""
    @abstractmethod
    def penalty(self, weights: np.ndarray) -> float:
        pass
    @abstractmethod
    def grad(self, weights: np.ndarray) -> np.ndarray:
        pass 