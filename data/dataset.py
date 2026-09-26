import torch
from typing import List, Tuple

class Solution:
    def batch_loader(self, raw_dataset: str, context_length: int, batch_size: int) -> Tuple[List[List[str]], List[List[str]]]:
        torch.manual_seed(0)
        data = raw_dataset.split(" ")
        start = torch.randint(len(data) - context_length, (batch_size,))
        X = [data[i:i + context_length] for i in start]
        Y = [data[i + 1: i + 1 +context_length] for i in start]
        return X, Y
