import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        # Start with the input layer
        activation = x
        
        activation = x
        num_layers = len(weights)
        # Iterate through each layer in the network
        for i in range(num_layers-1):
            # 1. Linear Transformation: z = x * W + b
            z = np.dot(activation, weights[i]) + biases[i]
            # Activation Function: Apply ReLU (except for the last layer)
            activation = np.maximum(0, z)
        activation = np.dot(activation, weights[-1]) + biases[-1]
        return np.round(activation, 5)


