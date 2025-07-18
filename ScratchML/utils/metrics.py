import numpy as np
from typing import Callable, Dict

# Словарь для регистрации кастомных метрик
_METRICS: Dict[str, Callable] = {}

def register_metric(name: str, func: Callable):
    """Регистрирует новую метрику по имени."""
    _METRICS[name] = func

def get_metric(name: str) -> Callable:
    if name not in _METRICS:
        raise ValueError(f"Метрика '{name}' не найдена.")
    return _METRICS[name]

# --- Классификация ---
def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(y_true == y_pred)

register_metric('accuracy', accuracy)

def precision(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

register_metric('precision', precision)

def recall(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

register_metric('recall', recall)

def f2(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    beta2 = 4
    denom = beta2 * p + r
    return (1 + beta2) * p * r / denom if denom > 0 else 0.0

register_metric('f2', f2)

# --- Регрессия ---
def rmse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

register_metric('rmse', rmse)

def r2(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    u = ((y_true - y_pred) ** 2).sum()
    v = ((y_true - y_true.mean()) ** 2).sum()
    return 1 - u / v if v != 0 else 0.0

register_metric('r2', r2) 