import numpy as np
from numpy.typing import NDArray
import math


class Solution:
    def sigmoid(self, x: float) -> float:
        return 1 / (1 + math.exp(-x))
    
    def relu(self, x: NDArray[np.float64]) -> float:
        return max(0.0, x)

    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # x: 1D input array
        # w: 1D weight array (same length as x)
        # b: scalar bias
        # activation: "sigmoid" or "relu"
        #
        # Pre-activation: z = dot(x, w) + b
        # Sigmoid: σ(z) = 1 / (1 + exp(-z))
        # ReLU: max(0, z)
        # return round(your_answer, 5)
        model_prediction = np.dot(x, w) + b
        if activation == 'sigmoid':
            return np.round(self.sigmoid(model_prediction), 5)
        else:
            return np.round(self.relu(model_prediction), 5)

