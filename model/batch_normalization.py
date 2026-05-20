import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        # 1. Convert all incoming lists to NumPy arrays
        x_arr = np.array(x)
        gamma_arr = np.array(gamma)
        beta_arr = np.array(beta)
        rmean_arr = np.array(running_mean)
        rvar_arr = np.array(running_var)

        if training:
            # 2. Compute current batch statistics
            mean = np.mean(x_arr, axis=0)
            variance = np.var(x_arr, axis=0)

            # 3. Normalize current batch features
            x_hat = (x_arr - mean) / np.sqrt(variance + eps)

            # 4. Update tracking history using momentum
            rmean_arr = (1 - momentum) * rmean_arr + momentum * mean
            rvar_arr = (1 - momentum) * rvar_arr + momentum * variance
        else:
            # 5. During inference, bypass batch stats and use history
            x_hat = (x_arr - rmean_arr) / np.sqrt(rvar_arr + eps)

        # 6. Apply scale (gamma) and shift (beta)
        y = gamma_arr * x_hat + beta_arr

        # 7. Round and cast back to native Python lists
        y_rounded = np.round(y, 4).tolist()
        rmean_rounded = np.round(rmean_arr, 4).tolist()
        rvar_rounded = np.round(rvar_arr, 4).tolist()

        return y_rounded, rmean_rounded, rvar_rounded
