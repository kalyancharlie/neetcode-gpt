class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places
        x = init
        ans = x
        while iterations:
            slope = 2 * x
            # If slope is negative, subtracting it adds to x (moves right).
            # If slope is positive, subtracting it reduces x (moves left).
            x = x - (learning_rate * slope)
            iterations -= 1
        return round(x, 5)
