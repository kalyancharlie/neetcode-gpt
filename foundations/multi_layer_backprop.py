import numpy as np
from typing import List

class Solution:
    def model_prediction(self, model_input: np.ndarray, model_weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
        """Calculate Linear Layer Output: z = x * W + b"""
        return np.dot(model_weights, model_input) + bias
    
    def relu(self, model_input: np.ndarray) -> np.ndarray:
        """Element-wise ReLU Activation Function"""
        return np.maximum(0, model_input)
    
    def get_error(self, model_prediction: np.ndarray, ground_truth: np.ndarray) -> float:
        """Compute Mean Squared Error (MSE)"""
        diff = model_prediction - ground_truth
        return float(np.mean(np.square(diff)))
    
    def forward_and_backward(self,
                             x: List[float],
                             W1: List[List[float]], b1: List[float],
                             W2: List[List[float]], b2: List[float],
                             y_true: List[float]) -> dict:
        
        # 1. Convert all standard Python lists to NumPy arrays for matrix math
        x_arr = np.array(x)
        W1_arr = np.array(W1)
        b1_arr = np.array(b1)
        W2_arr = np.array(W2)
        b2_arr = np.array(b2)
        y_true_arr = np.array(y_true)
        
        # --- FORWARD PASS ---
        # Layer 1: Linear Transformation -> ReLU Activation
        h1 = self.model_prediction(x_arr, W1_arr, b1_arr)
        a1 = self.relu(h1)
        
        # Layer 2: Final Linear Transformation (Output Predictions)
        z2 = self.model_prediction(a1, W2_arr, b2_arr)
        
        # Calculate MSE Loss
        loss = self.get_error(z2, y_true_arr)
        
        # --- BACKWARD PASS (Backpropagation via Chain Rule) ---
        # Step 1: Gradient of Loss w.r.t Output Layer Output (z2)
        # dL/dz2 = 2/N * (z2 - y_true)
        dz2 = (2.0 / y_true_arr.size) * (z2 - y_true_arr)
        
        # Step 2: Gradients for Layer 2 Weights and Biases
        # dW2 is the outer product of incoming activations (a1) and error (dz2)
        dW2 = np.outer(dz2, a1)
        db2 = dz2
        
        # Step 3: Backpropagate the error through Layer 2 to Layer 1 Activations
        da1 = np.dot(dz2, W2_arr)
        
        # Step 4: Gradient through the ReLU activation function
        # Derivative is 1 if h1 > 0, otherwise 0
        dh1 = da1 * (h1 > 0)
        
        # Step 5: Gradients for Layer 1 Weights and Biases
        dW1 = np.outer(dh1, x_arr)
        db1 = dh1
        
        # 2. Return results converted back to Python lists, rounded to 4 decimals
        return {
            "loss": round(loss, 4),
            "dW1": (np.round(dW1, 4) + 0.0).tolist(),
            "db1": (np.round(db1, 4) + 0.0).tolist(),
            "dW2": (np.round(dW2, 4) + 0.0).tolist(),
            "db2": (np.round(db2, 4) + 0.0).tolist()
        }