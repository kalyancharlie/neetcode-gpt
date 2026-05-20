import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        # 1. Set the random seed for reproducibility
        torch.manual_seed(0)
        
        # 2. Calculate the standard deviation using the Xavier formula
        sigma = math.sqrt(2 / (fan_in + fan_out))
        
        # 3. Generate the (fan_out x fan_in) matrix scaled by sigma
        tensor = torch.randn(fan_out, fan_in) * sigma
        
        # 4. Convert to list and round every element to 4 decimal places
        nested_list = tensor.tolist()
        return [[round(val, 4) for val in row] for row in nested_list]


    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        # 1. Set the random seed for reproducibility
        torch.manual_seed(0)
        
        # 2. Calculate the standard deviation using the Kaiming formula
        sigma = math.sqrt(2 / fan_in)
        
        # 3. Generate the (fan_out x fan_in) matrix scaled by sigma
        tensor = torch.randn(fan_out, fan_in) * sigma
        
        # 4. Convert to list and round every element to 4 decimal places
        nested_list = tensor.tolist()
        return [[round(val, 4) for val in row] for row in nested_list]

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # Forward random input through num_layers with the given init_type.
        # Use torch.manual_seed(0) once at the start.
        # Return the std of activations after each layer, rounded to 2 decimals.
        # # 1. Set seed once at the very start
        # torch.manual_seed(0)

        # # 2. Initialize our random input tensor
        # x = torch.randn(1000, input_dim)
        
        # stds = []

        # # 3. Loop through each layer
        # for i in range(num_layers):
        #     fan_in = input_dim if i == 0 else hidden_dim
        #     fan_out = hidden_dim

        #     # 4. Use the specific class initialization methods to get the weight matrix
        #     if init_type == "xavier":
        #         sigma = math.sqrt(2 / (fan_in + fan_out))
        #     elif init_type == "kaiming":
        #         sigma = math.sqrt(2 / fan_in)
        #     else:
        #         sigma = 1.0

        #     # 5. Convert the nested Python list back into a PyTorch Tensor
        #     W = torch.randn(fan_out, fan_in) * sigma

        #     # 6. Matrix multiplication (Linear combination)
        #     z = torch.matmul(x, W.t())

        #     # 7. Activation function (ReLU)
        #     x = torch.relu(z)

        #     # 8. Calculate standard deviation and round to 2 decimals
        #     current_std = x.std().item()
        #     stds.append(round(current_std, 2))
        # return stds

        torch.manual_seed(0)
        dims = [input_dim] + [hidden_dim] * num_layers
        weights = []
        
        for i in range(num_layers):
            if init_type == "kaiming":
                std = math.sqrt(2.0 / dims[i])
            elif init_type == "xavier":
                std = math.sqrt(2.0 / (dims[i] + dims[i + 1]))
            elif init_type == "random":
                std = 1.0  # plain N(0,1), no scaling

            w = torch.randn(dims[i+1], dims[i]) * std
            weights.append(w)

        x = torch.randn(1, input_dim)
        stds = []
        for w in weights:
            x = x @ w.T
            x = torch.relu(x)
            stds.append(round(x.std().item(), 2))

        return stds
