import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_derivative(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64], N: int, X: NDArray[np.float64], desired_weight: int) -> float:
        # note that N is just len(X)
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.squeeze(np.matmul(X, weights))

    learning_rate = 0.01

    def train_model(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        num_iterations: int,
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # For each iteration:
        #   1. Compute predictions with get_model_prediction(X, weights)
        #   2. For each weight index j, compute gradient with get_derivative()
        #   3. Update: weights[j] -= learning_rate * gradient
        # Return np.round(final_weights, 5)
        # print(f"X: {X}")
        # print(f"Y: {Y}")
        # print(f"initial_weights: {initial_weights}")
        weights = initial_weights.copy()
        N = len(X)
        num_features = len(weights)

        while num_iterations:
            # 1. Find Model prediction with the current state of weights
            model_prediction = self.get_model_prediction(X, weights)

            # 2. Find the gradient for all the weights
            gradients = np.zeros(num_features)
            # This would tell us how to adjust our weights (whether to move to the left or right to ultimately reach global minima)
            for ind in range(len(weights)):
                gradients[ind] = self.get_derivative(model_prediction, Y, N, X, ind)

            # 3. Update all weights
            weights -= self.learning_rate * gradients
            num_iterations -= 1
        
        return np.round(weights, 5)

