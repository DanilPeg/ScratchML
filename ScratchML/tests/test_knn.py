import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_blobs, make_friedman1
from sklearn.model_selection import train_test_split
from ScratchML.neighbors.knn import KNNClassifier, KNNRegressor, gaussian_kernel
from ScratchML.utils.metrics import accuracy, rmse

X, y = make_moons(n_samples=300, noise=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

plt.figure(figsize=(12, 5))
for i, (name, kernel) in enumerate({"Обычный KNN": None, "Гауссово ядро": gaussian_kernel}.items()):
    clf = KNNClassifier(n_neighbors=15, kernel=kernel, kernel_params={"sigma": 0.5} if kernel else None)
    clf.fit(X_train, y_train)
    acc = accuracy(y_test, clf.predict(X_test))
    plt.subplot(1, 2, i+1)
    xx, yy = np.meshgrid(np.linspace(X[:,0].min()-0.5, X[:,0].max()+0.5, 200),
                         np.linspace(X[:,1].min()-0.5, X[:,1].max()+0.5, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = clf.predict(grid).reshape(xx.shape)
    plt.contourf(xx, yy, preds, alpha=0.2, levels=[-0.5,0.5,1.5], cmap='bwr')
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr', alpha=0.6, label='train')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='bwr', marker='x', label='test')
    plt.title(f"{name}\nAcc: {acc:.2f}")
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
plt.tight_layout()
plt.show()

X, y = make_blobs(n_samples=300, centers=3, n_features=2, cluster_std=2.0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

plt.figure(figsize=(12, 5))
for i, (name, kernel) in enumerate({"Обычный KNN": None, "Гауссово ядро": gaussian_kernel}.items()):
    clf = KNNClassifier(n_neighbors=10, kernel=kernel, kernel_params={"sigma": 1.0} if kernel else None)
    clf.fit(X_train, y_train)
    acc = accuracy(y_test, clf.predict(X_test))
    plt.subplot(1, 2, i+1)
    xx, yy = np.meshgrid(np.linspace(X[:,0].min()-1, X[:,0].max()+1, 200),
                         np.linspace(X[:,1].min()-1, X[:,1].max()+1, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = clf.predict(grid).reshape(xx.shape)
    plt.contourf(xx, yy, preds, alpha=0.2, levels=np.arange(4)-0.5, cmap='tab10')
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='tab10', alpha=0.6, label='train')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='tab10', marker='x', label='test')
    plt.title(f"{name}\nAcc: {acc:.2f}")
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
plt.tight_layout()
plt.show()

X, y = make_friedman1(n_samples=300, n_features=5, noise=5.0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_vis = X_test[:, [0]]
X_vis_grid = np.linspace(X_vis.min(), X_vis.max(), 200).reshape(-1, 1)

plt.figure(figsize=(12, 5))
for i, (name, kernel) in enumerate({"Обычный KNN": None, "Гауссово ядро": gaussian_kernel}.items()):
    reg = KNNRegressor(n_neighbors=10, kernel=kernel, kernel_params={"sigma": 1.0} if kernel else None)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    rmse_val = rmse(y_test, y_pred)
    plt.subplot(1, 2, i+1)
    y_grid_pred = reg.predict(np.hstack([X_vis_grid, np.zeros((len(X_vis_grid), X_test.shape[1]-1))]))
    plt.scatter(X_vis[:, 0], y_test, label='test', alpha=0.6)
    plt.plot(X_vis_grid[:, 0], y_grid_pred, color='red', label='KNN prediction')
    plt.title(f"{name}\nRMSE: {rmse_val:.2f}")
    plt.xlabel('x0')
    plt.ylabel('y')
    plt.legend()
plt.tight_layout()
plt.show() 