import torch
import torch.nn as nn
from typing import List


class Solution:

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:

        dead_fraction = []

        with torch.no_grad():

            for layer in model:
                x = layer(x)

                if isinstance(layer, nn.ReLU):
                    dead = (x == 0).all(dim = 0).float().mean().item()
                    dead_fraction.append(round(dead, 4))
            
        return dead_fraction

    def suggest_fix(self, dead_fractions: List[float]) -> str:

        if len(dead_fractions) == 0:
            return 'healthy'

        for fraction in dead_fractions:
            if fraction > 0.5:
                return 'use_leaky_relu'
        
        if dead_fractions[0] > 0.3:
            return 'reinitialize'

        increasing = all(dead_fractions[i] < dead_fractions[i + 1]
        for i in range(len(dead_fractions) - 1))

        if increasing and dead_fractions[-1] > 0.1:
            return 'reduce_learning_rate'

        return 'healthy'