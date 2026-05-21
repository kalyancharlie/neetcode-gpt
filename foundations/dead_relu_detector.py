import torch
import torch.nn as nn
from typing import List


class Solution:

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        # Forward pass through the model.
        # After each ReLU layer, compute the fraction of neurons that are dead.
        # A neuron is dead if it outputs 0 for ALL samples in the batch.
        # Return a list of dead fractions (one per ReLU layer), rounded to 4 decimals.
        out = x
        dead_fractions = []
        for layer in model.children():
            out = layer(out)
            if isinstance(layer, nn.ReLU):
                is_dead_per_neuron = (out <= 0).all(dim=0)
                dead_frac_val = round(is_dead_per_neuron.float().mean().item(), 4)
                dead_fractions.append(dead_frac_val)
        return dead_fractions
        

    def suggest_fix(self, dead_fractions: List[float]) -> str:
        # Given dead fractions per ReLU layer, suggest a fix.
        # Check in this order:
        # 1. 'use_leaky_relu' if any layer has dead fraction > 0.5
        # 2. 'reinitialize' if the first layer has dead fraction > 0.3
        # 3. 'reduce_learning_rate' if dead fraction strictly increases
        #    with depth AND the last layer's fraction > 0.1
        # 4. 'healthy' if max dead fraction < 0.1
        # 5. 'healthy' otherwise
        # If the list is empty, we can't analyze it
        if not dead_fractions:
            return 'healthy'
            
        # 1. 'use_leaky_relu' if ANY layer has dead fraction > 0.5
        if any(fraction > 0.5 for fraction in dead_fractions):
            return 'use_leaky_relu'
            
        # 2. 'reinitialize' if the FIRST layer has dead fraction > 0.3
        if dead_fractions[0] > 0.3:
            return 'reinitialize'
            
        # 3. 'reduce_learning_rate' if dead fraction strictly increases 
        # with depth AND the last layer's fraction > 0.1
        # (We check if every layer is strictly greater than the previous one)
        strictly_increasing = all(
            dead_fractions[i] > dead_fractions[i - 1] 
            for i in range(1, len(dead_fractions))
        )
        if strictly_increasing and dead_fractions[-1] > 0.1:
            return 'reduce_learning_rate'
            
        # 4. 'healthy' if MAX dead fraction < 0.1
        if max(dead_fractions) < 0.1:
            return 'healthy'
            
        # 5. 'healthy' otherwise
        return 'healthy'
