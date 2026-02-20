import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from .. import BaseRegression

X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


class LinearRegression(BaseRegression):
    def _approximation(self, X, w, b):
        return np.dot(X, w) + b.bias

    def _predict(self, X, w, b):
        return np.dot(X, w) + b


def mse(y_true, y_predicted):
    return np.mean((y_true - y_predicted) ** 2)


regressor = LinearRegression(lr=0.01)
regressor.fit(X_train, y_train)
predicted = regressor.predict(X_test)

mse_value = mse(y_test, predicted)
print(mse_value)

y_pred_line = regressor.predict(X)
cmap = plt.get_cmap("viridis")
fig = plt.figure(figsize=(8, 6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
plt.plot(X, y_pred_line, color="black", linewidth=2, label="Prediction")
plt.show()
