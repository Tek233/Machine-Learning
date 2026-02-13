import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(X_train.shape)
print(y_train.shape)

plt.scatter(X, y, marker="o", s=30)
plt.show()


class LinearRegression:
    def __init__(self, lr=0.001, n_iters=1000, marker="o", s=30):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass
