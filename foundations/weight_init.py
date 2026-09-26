import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)
        std = (2/(fan_out +fan_in))**0.5
        weights = torch.randn(fan_out, fan_in) * std
        return torch.round(weights, decimals=4).tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)
        std = (2/(fan_in))**0.5
        weights = torch.randn(fan_out, fan_in) * std
        return torch.round(weights, decimals=4).tolist()
        pass

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        torch.manual_seed(0)

        # Шаг 1: заранее собираем список размеров каждого "участка трубы"
        # dims[0] = input_dim, dims[1..num_layers] = hidden_dim
        dims = [input_dim] + [hidden_dim] * num_layers

        # Шаг 2: СНАЧАЛА создаём ВСЕ веса для всех слоёв, ничего ещё не прогоняя
        weights = []
        for i in range(num_layers):
            if init_type == 'xavier':
                std = math.sqrt(2.0 / (dims[i] + dims[i + 1]))
            elif init_type == 'kaiming':
                std = math.sqrt(2.0 / dims[i])
            else:
                std = 1.0
            w = torch.randn(dims[i + 1], dims[i]) * std
            weights.append(w)

        # Шаг 3: теперь, когда веса уже готовы, прогоняем данные через них по очереди
        x = torch.randn(1, input_dim)
        stds = []
        for w in weights:
            x = x @ w.T
            x = torch.relu(x)
            stds.append(round(x.std().item(), 2))

        return stds
