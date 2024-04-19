import torch
import torch.nn as nn
from torchtyping import TensorType


class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        pass
        # Define the architecture here
        self.dense_1 = nn.Linear(28 * 28, 512)
        self.relu_1 = nn.ReLU()
        self.dropout_1 = nn.Dropout(p=0.2)
        self.dense_2 = nn.Linear(512, 10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        x = self.dense_1(images)
        x = self.relu_1(x)
        x = self.dropout_1(x)
        x = self.dense_2(x)
        x = self.sigmoid(x)
        # Return the model's prediction to 4 decimal places
        print(x)
        return torch.round(x, decimals=4)
