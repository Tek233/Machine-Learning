from collections import Counter

import numpy as np
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split

from helpers import euclidean_distance

cmap = ListedColormap(["red", "green", "blue"])


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_nearest_neighbors = np.argsort(distances)[: self.k]
        k_nearest_labels = [self.y_train[i] for i in k_nearest_neighbors]
        most_common_class = Counter(k_nearest_labels).most_common(1)
        return most_common_class[0][0]


iris = datasets.load_iris()
X, y = iris.data, iris.target
# print(iris.feature_names)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=48
)
# print(X_train.shape)
# print(y_train.shape)
# plt.figure()
# plt.scatter(
#     X[:, 0],
#     X[:, 1],
#     c=y,
#     cmap=cmap,
# )
# plt.show()
clf = KNN(k=3)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print(predictions)
accuracy = np.sum(predictions == y_test) / len(y_test)
print(accuracy)
