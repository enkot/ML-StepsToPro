import numpy as np

from numpy import ndarray as NumpyArray
from utils import min_max_scaler
from typing import List


class LinearRegressor:
    def fit(self, X: NumpyArray, y: NumpyArray):
        X = np.column_stack((np.ones(len(X)), X))
        theta, hypothesisHistory = self.gradient_descent(X, y)
        self.theta = theta
        self.hypothesisHistory = hypothesisHistory
        return X
    
    def predict(self, X: NumpyArray):
        X = np.c_[np.ones(len(X)), X]
        return X @ self.theta

    def compute_cost(self, X, y, theta):
        m = len(y)
        return 1 / (2 * m) * sum((X @ theta - y) ** 2)

    def gradient_descent(
            self,
            X: NumpyArray,
            y: NumpyArray,
            alpha: float = 0.2, 
            num_iters: int = 400) -> (NumpyArray, List):
        m = len(y)
        theta = np.zeros(X.shape[1])
        hypothesisHistory = []

        for _ in range(1, num_iters):
            hypothesis = X @ theta
            theta = theta - alpha * 1 / m * X.T @ (hypothesis - y)
            hypothesisHistory.append(hypothesis)

        return theta, hypothesisHistory
