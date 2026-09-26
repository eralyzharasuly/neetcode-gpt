import torch
from torchtyping import TensorType
from typing import Tuple

class Solution:
    def create_batches(self, data: TensorType[int], context_length: int, batch_size: int) -> Tuple[TensorType[int], TensorType[int]]:
        torch.manual_seed(0)
        start = torch.randint(len(data) - context_length, (batch_size,))
        X = torch.stack([data[i:i + context_length] for i in start])
        Y = torch.stack([data[i + 1: i + 1 +context_length] for i in start])
        return X, Y