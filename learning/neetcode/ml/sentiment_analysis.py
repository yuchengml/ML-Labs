import torch
import torch.nn as nn
from torchtyping import TensorType


class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)
        self.embed = torch.nn.Embedding(vocabulary_size, 16)
        self.linear = torch.nn.Linear(16, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x: TensorType[int]) -> TensorType[float]:
        # Hint: The embedding layer outputs a B, T, embed_dim tensor
        # but you should average it into a B, embed_dim tensor before using the Linear layer

        # Return a B, 1 tensor and round to 4 decimal places
        # Input x: (2, 2)
        # print(x.shape).
        embed_x = torch.mean(self.embed(x), 1)  # After embedding: (2, 2, 16) -> After reduce_mean: (2, 16)
        # print(embed_x.shape)
        x = self.linear(embed_x)  # (2, 1)
        # print(x.shape)
        x = self.sigmoid(x)
        return torch.round(x, decimals=4)
