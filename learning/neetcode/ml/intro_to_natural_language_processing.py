import torch
import torch.nn as nn
from torchtyping import TensorType


# torch.tensor(python_list) returns a Python list as a tensor
class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        word_set = set()
        for s in positive:
            word_set = word_set.union(set(s.split(' ')))
        for s in negative:
            word_set = word_set.union(set(s.split(' ')))
        word_list = sorted(list(word_set))
        # print(word_set)

        index_ids = []
        for s in positive + negative:
            index_ids.append(torch.tensor([word_list.index(w) + 1 for w in s.split(' ')]))

        return torch.nn.utils.rnn.pad_sequence(index_ids, batch_first=True)
