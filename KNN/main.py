import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

cmap = ListedColormap(["red", "green", "blue"])
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X, y = iris.data, iris.target
print(iris.feature_names)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=48
)
print(X_train.shape)
print(y_train.shape)
plt.figure()
plt.scatter(
    X[:, 0],
    X[:, 1],
    c=y,
    cmap=cmap,
)
plt.show()
