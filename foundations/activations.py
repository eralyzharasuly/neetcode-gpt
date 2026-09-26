import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.round( 1 / ( 1 + np.exp(-z)), 5 )
        pass

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.array([max(0.0, x) for x in z])
        pass
