import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from ScratchML.linear.logistic_regression import LogisticRegression
from ScratchML.optim.sgd import SGD
from ScratchML.optim.momentum import Momentum
from ScratchML.optim.adam import Adam
from ScratchML.regularization.l2 import L2
from ScratchML.utils.metrics import accuracy

np.random.seed(42)
X, y = make_classification(n_samples=300, n_features=2, n_redundant=0, n_informative=2,
                           n_clusters_per_class=1, class_sep=1.5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

optimizers = {
    'SGD': SGD(lr=0.1),
    'Momentum': Momentum(lr=0.1, momentum=0.9),
    'Adam': Adam(lr=0.1)
}

plt.figure(figsize=(15, 5))
for i, (name, opt) in enumerate(optimizers.items()):
    model = LogisticRegression(optimizer=opt, regularizer=L2(alpha=0.01), n_iter=100, batch_size=32, fit_intercept=True, random_state=42)
    X_ = np.copy(X_train)
    y_ = np.copy(y_train)
    n_samples = X_.shape[0]
    losses = []
    model.optimizer = opt
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
            z = X_batch @ w
            p = 1 / (1 + np.exp(-z))
            grad = X_batch.T @ (p - y_batch) / len(y_batch)
            grad[1:] += 0.01 * w[1:]  # L2
            w = opt.update(w, grad)
        # Логлосс на всей выборке
        z_full = Xb @ w
        p_full = 1 / (1 + np.exp(-z_full))
        eps = 1e-8
        loss = -np.mean(y_ * np.log(p_full + eps) + (1 - y_) * np.log(1 - p_full + eps))
        losses.append(loss)
    # Сохраняем веса
    model.intercept_ = w[0]
    model.coef_ = w[1:]
    model.classes_ = np.unique(y_train)  #  predict/predict_proba не работает..
    plt.subplot(2, len(optimizers), i+1)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr', alpha=0.5, label='train')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='bwr', marker='x', label='test')
    xx, yy = np.meshgrid(np.linspace(X[:,0].min()-1, X[:,0].max()+1, 200),
                         np.linspace(X[:,1].min()-1, X[:,1].max()+1, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = model.predict_proba(grid)[:, 1].reshape(xx.shape)
    plt.contourf(xx, yy, probs, levels=[0, 0.5, 1], alpha=0.2, colors=['blue', 'red'])
    plt.title(f"{name}\nAcc: {accuracy(y_test, model.predict(X_test)):.2f}")
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.subplot(2, len(optimizers), len(optimizers)+i+1)
    plt.plot(losses)
    plt.title(f"{name} loss")
    plt.xlabel('Эпоха')
    plt.ylabel('LogLoss')
plt.tight_layout()
plt.show()

from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=300, centers=3, n_features=2, cluster_std=2.0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression(optimizer=Adam(lr=0.1), regularizer=L2(alpha=0.01), n_iter=150, batch_size=32, fit_intercept=True, random_state=42, multi_class='ovr')
model.fit(X_train, y_train)
acc = accuracy(y_test, model.predict(X_test))

plt.figure(figsize=(7, 5))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='tab10', alpha=0.5, label='train')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='tab10', marker='x', label='test')
xx, yy = np.meshgrid(np.linspace(X[:,0].min()-1, X[:,0].max()+1, 200),
                     np.linspace(X[:,1].min()-1, X[:,1].max()+1, 200))
grid = np.c_[xx.ravel(), yy.ravel()]
preds = model.predict(grid).reshape(xx.shape)
plt.contourf(xx, yy, preds, alpha=0.2, levels=np.arange(4)-0.5, cmap='tab10')
plt.title(f"Multiclass (OvR)\nAcc: {acc:.2f}")
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.tight_layout()
plt.show() 