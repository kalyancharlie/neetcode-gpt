import numpy as np
from numpy.typing import NDArray
import math


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # def calc(num):
        #     return 1 / (1 + math.exp(-num))
        # return np.round(np.array([calc(num) for num in z]), 5)
        # OR
        return np.round(1 / (1 + np.exp(-z)), 5)

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        # return np.array([max(0, num) for num in z])
        # OR
        return np.maximum(0, z)
