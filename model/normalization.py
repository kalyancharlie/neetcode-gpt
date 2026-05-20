import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], gamma: NDArray[np.float64], beta: NDArray[np.float64]) -> NDArray[np.float64]:
        # x: 1D feature vector
        # gamma: 1D scale parameter (same length as x)
        # beta: 1D shift parameter (same length as x)
        # eps = 1e-5
        # Normalize: x_hat = (x - mean) / sqrt(var + eps)
        # Scale and shift: out = gamma * x_hat + beta
        # return np.round(your_answer, 5)
        
        eps = 1e-5
        # Step 1: Calculate Mean & Variance of input
        mean = np.mean(x)
        variance = np.var(x)

        # Step 2: Normalize input to 0-mean and 1-variance
        x_hat = (x - mean)/np.sqrt(variance + eps) # (x-mu/std.)

        # Step 3: Scale &  Shift using learnable parameters (gamma & beta) otherwise network might not learn anything for some inputs
        out = gamma * x_hat + beta

        return np.round(out, 5)

