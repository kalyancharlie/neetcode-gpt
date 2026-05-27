import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        # Architecture: Linear(784, 512) -> ReLU -> Dropout(0.2) -> Linear(512, 10) -> Sigmoid
        self.linear = nn.Linear(784, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.linear2 = nn.Linear(512, 10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, images: TensorType[float]) -> TensorType[float]:
        # Input images is of MNIST dataset of 28x28x1 which is a grayscale image representation pixel intensity from black to white.
        # This 28x28x1 is flattend to single 1d array 28x28 = 784.
        # If its a standard color image (R,G,B) then it would of shape 28x28x3
        torch.manual_seed(0)
        # images shape: (batch_size, 784)
        # Return the model's prediction to 4 decimal places
        images = self.linear(images)
        images = self.relu(images)
        images = self.dropout(images) # To randomly turn off few neurons
        images = self.linear2(images)
        images = self.sigmoid(images)
        return torch.round(images, decimals=4)
