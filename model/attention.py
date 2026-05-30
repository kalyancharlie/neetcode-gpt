import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)

        self.attention_dim = attention_dim
        
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        self.key_proj = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.query_proj = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.value_proj = nn.Linear(embedding_dim, attention_dim, bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        # 4. Apply softmax(dim=2) to masked scores
        # 5. Return (scores @ V) rounded to 4 decimal places
        
        # 1. Project input through K, Q, V linear layers
        k = self.key_proj(embedded)
        q = self.query_proj(embedded)
        v = self.value_proj(embedded)
        
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # We use .transpose(-2, -1) to dynamically handle the sequence length 
        # and attention dimensions, preserving the batch dimension (if present).
        k_t = k.transpose(-2, -1)
        scores = torch.matmul(q, k_t) / math.sqrt(self.attention_dim)
        
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        # then masked_fill positions where mask == 0 with float('-inf')
        seq_len = scores.size(-1)
        # Create a lower-triangular matrix of ones
        mask = torch.tril(torch.ones(seq_len, seq_len, device=scores.device))
        # Apply the mask, filling upper-triangular elements with negative infinity
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 4. Apply softmax(dim=2) to masked scores
        attention_weights = torch.softmax(scores, dim=2)
        
        # 5. Return (scores @ V) rounded to 4 decimal places
        output = torch.matmul(attention_weights, v)
        return torch.round(output, decimals=4)
