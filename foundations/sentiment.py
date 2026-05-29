import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)
        
        # Maps vocabulary IDs to 16-dimensional vectors
        self.embedding = nn.Embedding(vocabulary_size, 16)
        
        # The input feature size must match the embedding dimension (16)
        self.linear = nn.Linear(16, 1)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: TensorType[int]) -> TensorType[float]:
        # Input x shape: (B, T)
        
        x = self.embedding(x)
        # Shape is now (B, T, 16)
        
        # FIX: Average across the sequence dimension (T is index 1)
        x = torch.mean(x, dim=1)
        # Shape is now (B, 16)
        
        x = self.linear(x)
        # Shape is now (B, 1)
        
        x = self.sigmoid(x)
        
        x = torch.round(x, decimals=4)
        
        return x