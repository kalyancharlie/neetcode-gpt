import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        stats = []
        out = x
        
        with torch.no_grad():
            for layer in model.children():
                # Forward pass through every layer (Linear, ReLU, etc.)
                out = layer(out)
                
                # Calculate the dead fraction properly based on batch behavior.
                if isinstance(layer, nn.Linear):
                    # We compute mean and std of the raw linear output
                    mean_val = round(out.mean().item(), 4)
                    std_val = round(out.std().item(), 4)
                    
                    # A neuron is "dead" if it outputs 0 for EVERY sample in the batch.
                    # We check if it's 0 along dim=0 (the batch dimension).
                    # Note: If it hasn't hit ReLU yet, we can check if it's <= 0 
                    # because a ReLU will turn all <= 0 values into dead entries.
                    is_dead_per_neuron = (out <= 0).all(dim=0)
                    dead_frac_val = round(is_dead_per_neuron.float().mean().item(), 4)
                    
                    stats.append({
                        'mean': mean_val,
                        'std': std_val,
                        'dead_fraction': dead_frac_val
                    })
                    
        return stats
            

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        stats = []
        
        # 1. Clear out any old gradients sitting in the model
        model.zero_grad()
        
        # 2. Forward pass: get the model's final prediction
        out = x
        for layer in model.children():
            out = layer(out)
            
        # 3. Calculate Mean Squared Error loss against the true target 'y'
        criterion = nn.MSELoss()
        loss = criterion(out, y)
        
        # 4. Backward pass: calculate gradients for all weights
        loss.backward()
        
        # 5. Collect stats from the nn.Linear weights
        for layer in model.children():
            if isinstance(layer, nn.Linear):
                # Access the gradient tensor of the layer's weights
                grad = layer.weight.grad
                if grad is not None:
                    stats.append({
                        "mean": round(grad.mean().item(), 4),
                        "std": round(grad.std().item(), 4),
                        "norm": round(grad.norm().item(), 4)
                    })
        return stats

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        # Priority 1: Check for Dead Neurons
        # If any linear layer has more than 90% dead activations
        for act in activation_stats:
            if act["dead_fraction"] > 0.5:
                return 'dead_neurons'
                
        # Priority 2: Check for Exploding Gradients
        # If any layer's gradient norm climbs over 50.0
        for grad in gradient_stats:
            if grad["norm"] > 10.0:
                return 'exploding_gradients'
                
        # Priority 3: Check for Vanishing Gradients
        # If any layer's gradient standard deviation drops below 1e-5 (0.00001)
        for grad in gradient_stats:
            if grad["std"] < 1e-4:
                return 'vanishing_gradients'
                
        return 'healthy'
