import numpy as np
import matplotlib.pyplot as plt
from ScratchML.linear.linear_regression import LinearRegression
from ScratchML.utils.metrics import rmse, r2
import sys
import os
sys.path.append(os.path.abspath("..")) 

np.random.seed(0)
X = np.linspace(0, 10, 100).reshape(-1, 1)
true_coef = 2.5
true_intercept = -1.0
y = true_coef * X[:, 0] + true_intercept + np.random.randn(100) * 1.0

model = LinearRegression(fit_intercept=True)
model.fit(X, y)
y_pred = model.predict(X)

plt.figure(figsize=(8, 5))
plt.scatter(X, y, label="Данные", alpha=0.7)
plt.plot(X, y_pred, color="red", label="Модель")
plt.xlabel("X")
plt.ylabel("y")
plt.title("LinearRegression: SVD")
plt.legend()
plt.show()

print(f'RMSE: {rmse(y, y_pred):.3f}')
print(f'R2: {r2(y, y_pred):.3f}')
print(f'Коэффициент: {model.coef_}')
print(f'Свободный член: {model.intercept_}')