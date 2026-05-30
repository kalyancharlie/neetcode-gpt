import torch
import torch.nn as nn
from torchtyping import TensorType

class MultiHeadedSelfAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int, num_heads: int):
        super().__init__()
        torch.manual_seed(0)
        # Create num_heads SingleHeadAttention instances using nn.ModuleList
        # Each head size = attention_dim // num_heads
        # Use: self.SingleHeadAttention(embedding_dim, head_size)
        # After the heads, add an output projection: nn.Linear(attention_dim, attention_dim, bias=False)
        # 1. Calculate the dimension size for each individual head
        head_size = attention_dim // num_heads
        
        # 2. Create the multiple attention heads. 
        # nn.ModuleList is required so PyTorch properly registers the sub-modules' parameters.
        self.heads = nn.ModuleList([
            self.SingleHeadAttention(embedding_dim, head_size) 
            for _ in range(num_heads)
        ])
        
        # 3. Create the final output projection matrix (W_O in the Attention Is All You Need paper)
        # This mixes the concatenated head outputs back into the expected attention_dim.
        self.output_proj = nn.Linear(attention_dim, attention_dim, bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # Run each head on the input, concatenate outputs along dim=2
        # Pass concatenated result through the output projection (W_O)
        # Return result rounded to 4 decimal places

        # 1. Process the input sequence through every independent attention head
        # Each head returns a tensor of shape: (batch_size, sequence_length, head_size)
        head_outputs = [head(embedded) for head in self.heads]
        
        # 2. Concatenate the outputs from all heads along the feature dimension (dim=2)
        # Resulting shape: (batch_size, sequence_length, attention_dim)
        concat_output = torch.cat(head_outputs, dim=2)
        
        # 3. Pass the concatenated tensor through the linear projection layer
        projected_output = self.output_proj(concat_output)
        
        # 4. Return the result rounded to 4 decimal places
        return torch.round(projected_output, decimals=4)

    class SingleHeadAttention(nn.Module):
        def __init__(self, embedding_dim: int, attention_dim: int):
            super().__init__()
            torch.manual_seed(0)
            self.key_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
            self.query_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
            self.value_gen = nn.Linear(embedding_dim, attention_dim, bias=False)

        def forward(self, embedded: TensorType[float]) -> TensorType[float]:
            k = self.key_gen(embedded)
            q = self.query_gen(embedded)
            v = self.value_gen(embedded)

            scores = q @ torch.transpose(k, 1, 2) # @ is the same as torch.matmul()
            context_length, attention_dim = k.shape[1], k.shape[2]
            scores = scores / (attention_dim ** 0.5)

            lower_triangular = torch.tril(torch.ones(context_length, context_length))
            mask = lower_triangular == 0
            scores = scores.masked_fill(mask, float('-inf'))
            scores = nn.functional.softmax(scores, dim = 2)

            return scores @ v
