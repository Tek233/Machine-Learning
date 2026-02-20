import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from .. import BaseRegression

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


class LogisticRegression(BaseRegression):
    def _approximation(self, X, w, b):
        linear_model = np.dot(X, w) + b.bias
        return self._sigmoid(linear_model)

    def _predict(self, X, w, b):
        linear_model = np.dot(X, w) + b
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_cls

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_test)
    return accuracy


regressor = LogisticRegression(lr=0.01, n_iters=10000)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)
print(f"accuracy: {accuracy(y_test, predictions)}")
