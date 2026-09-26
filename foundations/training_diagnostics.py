import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        result = []

        with torch.no_grad():
            current = x

            for layer in model:
                current = layer(current)

                if isinstance(layer, nn.Linear):
                    mean_val = torch.mean(current).item()
                    std_val = torch.std(current).item()

                    dead = (current <= 0).all(dim = 0)
                    dead_fraction = dead.float().mean().item()

                    result.append({
                        'mean': round(mean_val, 4),
                        'std': round(std_val, 4),
                        'dead_fraction': round(dead_fraction, 4)
                    })
            
        return result
 
    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        model.zero_grad()
        y_pred = model(x)
        loss = nn.MSELoss()(y_pred, y)
        loss.backward()
        result = []

        for layer in model:
            if isinstance(layer, nn.Linear):
                grad = layer.weight.grad
                mean_val = round(grad.mean().item(), 4)
                std_val = round(grad.std().item(), 4)
                norm_val = round(torch.norm(grad).item(), 4)
                result.append({'mean': mean_val, 'std': std_val, 'norm': norm_val})

        return result

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        
        for s in activation_stats:
            if s['dead_fraction'] > 0.5:
                return 'dead_neurons'
        
        for s in gradient_stats:
            if s['norm'] > 1000:
                return 'exploding_gradients'
        
        if gradient_stats and gradient_stats[-1]['norm'] < 1e-5:
            return 'vanishing_gradients'

        for s in activation_stats:
            if s['std'] < 0.1:
                return 'vanishing_gradients'
            if s['std'] > 10.0:
                return 'exploding_gradients'
        
        return 'healthy'