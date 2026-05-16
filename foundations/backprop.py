import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:

    def model_prediction(self, model_input: NDArray[np.float64], model_weights: NDArray[np.float64], bias: float) -> NDArray[np.float64]:
        """Calculate Model Prediction"""
        return np.dot(model_input, model_weights) + bias
    
    def sigmoid(self, model_prediction: NDArray[np.float64]):
        """Sigmoid Activation Function"""
        return 1 / (1 + np.exp(-model_prediction))
    
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)

        # Step 1: Model Prediction Calculation -> z = x * w + b
        model_prediction = self.model_prediction(x, w, b)
        print(f'Model Prediction: {model_prediction}')
        # Step 2: Passing Model Prediction via Activation Function y_hat = sigmoid(z)
        y_hat = self.sigmoid(model_prediction)
        print(f'y_hat: {y_hat}')
        # Step 3: Loss/Error Calculation
        dZ = (y_hat - y_true) * y_hat * (1 - y_hat)
        dL_dw = dZ * x
        dL_db = dZ
        return np.round(dL_dw, 5), np.round(dL_db, 5)

