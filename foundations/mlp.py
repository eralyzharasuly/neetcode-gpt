import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        a = np.array(x)
        for i in range(len(weights)):
            W = np.array(weights[i])
            B = np.array(biases[i])
            z = np.dot(a, W) + B
            if i < len(weights) - 1:
                a = np.maximum(z, 0)
            else:
                a = z
        return np.round(a, 5)

        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        pass
