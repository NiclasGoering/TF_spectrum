import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Set, Tuple


class DeepNN(nn.Module):
    def __init__(self, d: int, hidden_size: int, depth: int, mode: str = 'standard', gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma
        
        torch.set_default_dtype(torch.float32)
        
        self.mode = mode
        self.depth = depth
        self.hidden_size = hidden_size
        self.input_dim = d
        
        layers = []
        prev_dim = d
        self.layer_lrs = []  # Store layerwise learning rates
        
        for layer_idx in range(depth):
            linear = nn.Linear(prev_dim, hidden_size)
            
            if mode == 'spectral':
                # Implement spectral initialization
                fan_in = prev_dim
                fan_out = hidden_size
                std = (1.0 / np.sqrt(fan_in)) * min(1.0, np.sqrt(fan_out / fan_in))
                nn.init.normal_(linear.weight, mean=0.0, std=std)
                nn.init.zeros_(linear.bias)
                self.layer_lrs.append(float(fan_out) / fan_in)
                
            elif mode == 'mup_no_align':
                # Weight scaling factors and initialization
                if layer_idx == 0:  # Embedding layer
                    scale = np.sqrt(hidden_size)  # n^{1/2}
                    std = 1.0 / np.sqrt(prev_dim)  # 1/n initialization
                    lr_scale = 1.0  # O(1) learning rate
                else:  # Hidden layers
                    scale = 1.0  # n^0 = 1
                    std = 1.0 / (2 * np.sqrt(prev_dim))  # 1/(2n) initialization
                    lr_scale = 1.0 / np.sqrt(prev_dim)  # O(1/√n) learning rate
                
                # Create and scale the layer
                linear.weight.data = scale * torch.randn(hidden_size, prev_dim) * std
                linear.bias.data.zero_()
                self.layer_lrs.append(lr_scale)
                
            else:  # standard
                nn.init.xavier_uniform_(linear.weight)
                nn.init.zeros_(linear.bias)
                self.layer_lrs.append(1.0)
            
            layers.extend([
                linear,
                nn.ReLU()
            ])
            prev_dim = hidden_size
        
        # Final layer
        final_layer = nn.Linear(prev_dim, 1)
        if mode == 'spectral':
            fan_in = prev_dim
            fan_out = 1
            std = (1.0 / np.sqrt(fan_in)) * min(1.0, np.sqrt(fan_out / fan_in))
            nn.init.normal_(final_layer.weight, std=std)
            self.layer_lrs.append(float(fan_out) / fan_in)
        elif mode == 'mup_no_align':
            scale = 1.0 / np.sqrt(hidden_size)  # n^{-1/2}
            std = 1.0 / 2  # 1/2 initialization
            lr_scale = 1.0 / np.sqrt(hidden_size)  # O(1/√n) learning rate
            final_layer.weight.data = scale * torch.randn(1, hidden_size) * std
            final_layer.bias.data.zero_()
            self.layer_lrs.append(lr_scale)
        else:
            nn.init.xavier_uniform_(final_layer.weight)
            self.layer_lrs.append(1.0)
            
        nn.init.zeros_(final_layer.bias)
        layers.append(final_layer)
        
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gamma = float(self.gamma)
        return self.network(x).squeeze() / gamma
    
    def get_layer_learning_rates(self, base_lr: float) -> List[float]:
        """Return list of learning rates for each layer"""
        return [base_lr * lr for lr in self.layer_lrs]