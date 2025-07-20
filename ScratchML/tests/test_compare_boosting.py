import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor as SklearnGBR
from ScratchML.ensemble.gradient_boosting import GradientBoostingRegressor
from ScratchML.utils.metrics import rmse

# --- СЛОЖНЫЙ ДАТАСЕТ: California housing ---
data = fetch_california_housing()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

params = dict(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)

models = [
    (GradientBoostingRegressor(**params), "ScratchML OrderedBoosting"),
    (SklearnGBR(**params), "sklearn GradientBoosting")
]

plt.figure(figsize=(12, 5))
for i, (model, name) in enumerate(models):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse_val = rmse(y_test, y_pred)
    plt.subplot(1, 2, i+1)
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title(f"{name}\nRMSE: {rmse_val:.3f}")
    plt.xlabel('y_test')
    plt.ylabel('y_pred')
plt.tight_layout()
plt.show() 