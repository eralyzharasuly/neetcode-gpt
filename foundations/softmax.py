import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        z = z - np.max(z)
        expon = np.exp(z)
        sum = np.sum(expon)
        result = expon/sum
        return np.round(expon/sum,4)
        pass
