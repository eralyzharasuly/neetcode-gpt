import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        w = np.zeros(np.shape(X[1]))
        b = 0.0

        for i in range(epochs):

            y_hat = X @ w + b
            loss = np.mean((y_hat - y)**2)
            
            dw = (2/len(y)) * X.T @ (y_hat - y)
            db = np.mean(2*(y_hat - y))

            w -= lr*dw
            b -= lr*db
        
        return (np.round(w, 5), round(b, 5))
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))
        pass
