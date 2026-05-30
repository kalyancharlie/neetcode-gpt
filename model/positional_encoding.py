import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_positional_encoding(self, seq_len: int, d_model: int) -> NDArray[np.float64]:
        # PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
        #
        # Hint: Use np.arange() to create position and dimension index vectors,
        # then compute all values at once with broadcasting (no loops needed).
        # Assign sine to even columns (PE[:, 0::2]) and cosine to odd columns (PE[:, 1::2]).
        # Round to 5 decimal places.
        
        # 1. Create an empty matrix of zeros with the correct shape
        PE = np.zeros((seq_len, d_model))
        
        # 2. Get the positions (rows) and reshape into a vertical column
        positions = np.arange(seq_len).reshape(seq_len, 1)
        
        # 3. Get the even dimension indices
        even_indices = np.arange(0, d_model, 2)
        
        # 4. Calculate the denominator: 10000^(2i / d_model)
        denominator = 10000 ** (even_indices / d_model)
        
        # 5. Divide the vertical positions by the horizontal denominators 
        # (NumPy broadcasting creates the full 2D grid automatically)
        angles = positions / denominator
        
        # 6. Apply sine to the even columns (0, 2, 4...)
        PE[:, 0::2] = np.sin(angles)
        
        # 7. Apply cosine to the odd columns (1, 3, 5...)
        PE[:, 1::2] = np.cos(angles)
        
        # 8. Round all values to 5 decimal places as required by the problem
        return np.round(PE, 5)
