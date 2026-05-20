import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        # Implement RMS Normalization (similar to LayerNorm but without mean centering or beta)
        # Normalize x, then scale by gamma
        # Return result rounded to 4 decimal places as a list
        # Convert inputs to numpy arrays for vector operations
        x_arr = np.array(x)
        gamma_arr = np.array(gamma)
        
        # 1. Square the values
        square = np.square(x_arr)
        
        # 2. Calculate the Root Mean Square (RMS) denominator
        rms = np.sqrt(np.mean(square) + eps)
        
        # 3. Normalize the input and scale by gamma
        out = (x_arr / rms) * gamma_arr
        
        # 4. Round to 4 decimal places and convert back to a standard list
        return np.round(out, 4).tolist()
