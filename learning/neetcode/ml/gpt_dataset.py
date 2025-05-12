import torch
from typing import List, Tuple


class Solution:
    def batch_loader(self, raw_dataset: str, context_length: int, batch_size: int) -> Tuple[List[List[str]]]:
        # You must start by generating batch_size different random indices in the appropriate range
        # using a single call to torch.randint()
        torch.manual_seed(0)
        tokens = raw_dataset.split(' ')

        indices = torch.randint(low=0, high=len(tokens) - context_length, size=(batch_size,)).tolist()
        X, Y = [], []
        for idx in indices:
            X.append(tokens[idx: idx + context_length])
            Y.append(tokens[idx + 1: idx + context_length + 1])
        # print(X, Y)

        return X, Y
