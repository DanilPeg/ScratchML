import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from ScratchML.linear.sgd_regressor import SGDRegressor
from ScratchML.optim.sgd import SGD
from ScratchML.optim.momentum import Momentum
from ScratchML.optim.nesterov import Nesterov
from ScratchML.optim.rmsprop import RMSProp
from ScratchML.optim.adadelta import AdaDelta
from ScratchML.optim.adam import Adam
from ScratchML.optim.nadam import Nadam
from ScratchML.regularization.l2 import L2

np.random.seed(42)
X, y = make_regression(n_samples=500, n_features=10, noise=30, random_state=42)
n_outliers = 20
outlier_idx = np.random.choice(X.shape[0], n_outliers, replace=False)
y[outlier_idx] += np.random.randn(n_outliers) * 200

optimizers = {
    'SGD': SGD(lr=0.001),
    'Momentum': Momentum(lr=0.001, momentum=0.9),
    'Nesterov': Nesterov(lr=0.001, momentum=0.9),
    'RMSProp': RMSProp(lr=0.001),
    'AdaDelta': AdaDelta(),
    'Adam': Adam(lr=0.001),
    'Nadam': Nadam(lr=0.001)
}

loss_histories = {}

for name, opt in optimizers.items():
    model = SGDRegressor(optimizer=opt, regularizer=L2(alpha=0.1), n_iter=100, batch_size=64, fit_intercept=True, random_state=42)
    X_ = np.copy(X)
    y_ = np.copy(y)
    n_samples = X_.shape[0]
    losses = []
    model.optimizer = opt  # сброс состояния оптимизатора
    model.coef_ = None
    model.intercept_ = None
    Xb = np.hstack([np.ones((n_samples, 1)), X_])
    w = np.random.RandomState(42).randn(Xb.shape[1]) * 0.01
    for epoch in range(model.n_iter):
        indices = np.random.RandomState(42 + epoch).permutation(n_samples)
        for start in range(0, n_samples, model.batch_size):
            end = min(start + model.batch_size, n_samples)
            batch_idx = indices[start:end]
            X_batch = Xb[batch_idx]
            y_batch = y_[batch_idx]
            y_pred = X_batch @ w
            grad = X_batch.T @ (y_pred - y_batch) / len(y_batch)
            if model.regularizer is not None:
                grad[1:] += model.regularizer.grad(w[1:])
            # Gradient clipping
            grad = np.clip(grad, -100, 100)
            w = opt.update(w, grad)
            # Защита от NaN/inf
            if not np.all(np.isfinite(w)):
                w = np.nan_to_num(w, nan=0.0, posinf=1e2, neginf=-1e2)
        y_pred_full = Xb @ w
        loss = np.mean((y_pred_full - y_) ** 2)
        losses.append(loss)
    loss_histories[name] = losses

plt.figure(figsize=(10, 6))
for name, losses in loss_histories.items():
    plt.plot(losses, label=name)
plt.xlabel("Эпоха")
plt.ylabel("MSE Loss")
plt.title("Сходимость лосса для разных оптимизаторов (SGDRegressor, сложный датасет)")
plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.show() 