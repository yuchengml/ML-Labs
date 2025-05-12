import torch
import torch.nn as nn
from torchtyping import TensorType


class MultiHeadedSelfAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int, num_heads: int):
        super().__init__()
        torch.manual_seed(0)
        # Hint: nn.ModuleList() will be useful. It works the same as a Python list
        # but is useful here since instance variables of any subclass of nn.Module
        # must also be subclasses of nn.Module

        # Use self.SingleHeadAttention(embedding_dim, head_size) to instantiate. You have to calculate head_size.
        self.multiheads = nn.ModuleList(
            [self.SingleHeadAttention(embedding_dim, attention_dim // num_heads) for i in range(num_heads)])

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # Return answer to 4 decimal places
        outputs = []
        for i, l in enumerate(self.multiheads):
            outputs.append(l(embedded))
        return torch.round(torch.cat(outputs, -1), decimals=4)

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

            scores = q @ torch.transpose(k, 1, 2)  # @ is the same as torch.matmul()
            context_length, attention_dim = k.shape[1], k.shape[2]
            scores = scores / (attention_dim ** 0.5)

            lower_triangular = torch.tril(torch.ones(context_length, context_length))
            mask = lower_triangular == 0
            scores = scores.masked_fill(mask, float('-inf'))
            scores = nn.functional.softmax(scores, dim=2)

            return scores @ v
