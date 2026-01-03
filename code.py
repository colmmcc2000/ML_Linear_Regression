import numpy as np

class LinearRegressionScratch:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.lr = learning_rate
        self.epochs = iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # 1. Initialize parameters (weights for features, bias for intercept)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # 2. Gradient Descent Loop
        for _ in range(self.epochs):
            # Linear Model: y_hat = Xw + b
            y_predicted = np.dot(X, self.weights) + self.bias

            # Calculate Gradients
            # dw = (1/n) * X^T * (y_hat - y)
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            # db = (1/n) * sum(y_hat - y)
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update Parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    