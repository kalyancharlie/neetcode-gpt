import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))

        # 1. Get dimensions from X
        n_samples, n_features = X.shape
        
        # 2. Initialize weights (w) as a column vector of zeros, and bias (b) as 0.0
        w = np.zeros((n_features, 1))
        b = 0.0
        
        # 3. Reshape y from a 1D array (n_samples,) to a 2D column vector (n_samples, 1)
        y = y.reshape(-1, 1)
        
        # 4. Gradient Descent Loop
        for i in range(epochs):
            # Predict: y_hat = X @ w + b
            y_hat = X @ w + b
            
            # Calculate the error vector
            error = y_hat - y
            
            # Calculate gradients for weights and bias
            dw = (2 / n_samples) * (X.T @ error)
            db = (2 / n_samples) * np.sum(error)
            
            # Update weights and bias by taking a step in the opposite direction of the gradient
            w = w - lr * dw
            b = b - lr * db
            print(f'W:{w}, b:{b}')
            
        # 5. Format the final output as requested in the comments
        # We flatten w back to a 1D array using .flatten() to match the expected return type
        return (np.round(w.flatten(), 5), round(float(b), 5))
