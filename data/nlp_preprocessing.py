import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:

        all_words = set()

        sentences  = positive + negative
        for sentence in sentences:
            for word in sentence.split():
                all_words.add(word)
        all_words = sorted(all_words)

        word_id = {word: i + 1 for i, word in enumerate(all_words)}

        T = max(len(sentence.split()) for sentence in sentences)

        result = []
        for sentence in sentences:
            words = sentence.split()
            ids = [word_id[word] for word in words]
            while len(ids) < T:
                ids.append(0)
            result.append(ids)
        
        return torch.tensor(result, dtype = float)