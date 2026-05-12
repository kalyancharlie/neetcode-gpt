import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Round to 5 decimal places
        print(f'x: {X}')
        print(f'weights: {weights}')
        prediction = weights * X
        return np.round(prediction.sum(axis=1), 5)

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        # Round to 5 decimal places
        print(f'model_prediction: {model_prediction}')
        print(f'ground_truth: {ground_truth}')
        diff = (ground_truth - model_prediction)
        squared_diff = np.square(diff)
        return round(squared_diff.mean(), 5)
