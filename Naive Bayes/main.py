import numpy as np

X = np.array(
    [
        [5.1, 3.5, 1.4, 0.2],
        [4.9, 3.0, 1.4, 0.2],
        [7.0, 3.2, 4.7, 1.4],
        [6.4, 3.2, 4.5, 1.5],
        [6.3, 3.3, 6.0, 2.5],
    ]
)

y = np.array([0, 0, 1, 1, 2])

print(np.unique(y, return_counts=True)[1])


class NaiveBayes:
    def __init__(self):
        self.means = {}
        self.variances = {}
        self.priors = {}

    def fit(self, X, y):
        for c in np.unique(y):
            X_c = X[y == c]
            self.priors[c] = X_c.shape[0] / X.shape[0]
            self.means[c] = np.mean(X_c, axis=0)
            self.variances[c] = np.var(X_c, axis=0) + 1e-9  # smoothing value
        print(f"perior: {self.priors}")
        print(f"mean: {self.means}")

    def predict(self, x):
        prob = {}
        for c in self.priors.keys():
            mean = self.means[c]
            var = self.variances[c]
            prior = np.log(self.priors[c])
            log_pdf = -0.5 * np.log(2 * np.pi * var) - ((x - mean) ** 2 / (2 * var))
            prob[c] = prior + np.sum(log_pdf)
        print(f"\nPrrrr: {prob}\n")
        return max(prob, key=prob.get)


cls = NaiveBayes()
cls.fit(X, y)
print(f"prediction: {cls.predict([6.3, 3.3, 6.0, 2.5])}")
print(f"prediction: {cls.predict([5.1, 4, 1, 0.7])}")
print(f"prediction: {cls.predict([6.9, 3.0, 5.5, 4.5])}")
