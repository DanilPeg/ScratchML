import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_friedman1
from sklearn.model_selection import train_test_split
from ScratchML.ensemble.gradient_boosting import GradientBoostingClassifier, GradientBoostingRegressor
from ScratchML.tree.cart import DecisionTreeClassifier, DecisionTreeRegressor
from ScratchML.utils.metrics import accuracy, rmse

# --- КЛАССИФИКАЦИЯ: make_moons ---
X, y = make_moons(n_samples=300, noise=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

plt.figure(figsize=(12, 6))
for i, (model, name) in enumerate([
    (DecisionTreeClassifier(max_depth=4, min_samples_leaf=5), "DecisionTree"),
    (GradientBoostingClassifier(n_estimators=30, learning_rate=0.2, max_depth=3, random_state=42, tree_params={"min_samples_leaf":5}), "OrderedBoosting")
]):
    model.fit(X_train, y_train)
    acc = accuracy(y_test, model.predict(X_test))
    plt.subplot(1, 2, i+1)
    xx, yy = np.meshgrid(np.linspace(X[:,0].min()-0.5, X[:,0].max()+0.5, 200),
                         np.linspace(X[:,1].min()-0.5, X[:,1].max()+0.5, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = model.predict(grid).reshape(xx.shape)
    plt.contourf(xx, yy, preds, alpha=0.2, levels=[-0.5,0.5,1.5], cmap='bwr')
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr', alpha=0.6, label='train')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='bwr', marker='x', label='test')
    plt.title(f"{name}\nAcc: {acc:.2f}")
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
plt.tight_layout()
plt.show()

X, y = make_friedman1(n_samples=300, n_features=5, noise=5.0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

plt.figure(figsize=(12, 6))
for i, (model, name) in enumerate([
    (DecisionTreeRegressor(max_depth=10, min_samples_leaf=5), "DecisionTree"),
    (GradientBoostingRegressor(n_estimators=30, learning_rate=0.2, max_depth=3, random_state=42, tree_params={"min_samples_leaf":5}), "OrderedBoosting")
]):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse_val = rmse(y_test, y_pred)
    plt.subplot(1, 2, i+1)
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title(f"{name}\nRMSE: {rmse_val:.2f}")
    plt.xlabel('y_test')
    plt.ylabel('y_pred')
plt.tight_layout()
plt.show()

X = np.linspace(0, 10, 200).reshape(-1, 1)
y = np.sin(X[:, 0]) + np.random.randn(200) * 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

plt.figure(figsize=(12, 5))
for i, (model, name) in enumerate([
    (DecisionTreeRegressor(max_depth=6, min_samples_leaf=2), "DecisionTree"),
    (GradientBoostingRegressor(n_estimators=30, learning_rate=0.2, max_depth=3, random_state=42, tree_params={"min_samples_leaf":2}), "OrderedBoosting")
]):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    plt.subplot(1, 2, i+1)
    plt.scatter(X_test[:, 0], y_test, label='test', alpha=0.6)
    plt.scatter(X_test[:, 0], y_pred, color='red', label='Prediction', s=10)
    plt.title(f"{name}\nRMSE: {rmse(y_test, y_pred):.2f}")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
plt.tight_layout()
plt.show() 