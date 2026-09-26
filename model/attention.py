import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        self.key = nn.Linear(embedding_dim, attention_dim, bias = False)
        self.query = nn.Linear(embedding_dim, attention_dim, bias = False)
        self.value = nn.Linear(embedding_dim, attention_dim, bias = False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        k = self.key(embedded)
        q = self.query(embedded)
        v = self.value(embedded)

        score = (q @ torch.transpose(k,1,2))/(k.shape[-1]**0.5)

        seq_len = embedded.shape[1]
        mask = torch.tril(torch.ones(seq_len, seq_len))
        score = score.masked_fill(mask == 0, float('-inf'))

        attention_weights = nn.functional.softmax(score, dim = -1)

        output = attention_weights @ v

        return torch.round(output, decimals = 4)