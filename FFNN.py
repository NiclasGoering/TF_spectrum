import numpy as np
import torch
import torch.nn as nn
from typing import List

class DeepNN(nn.Module):
    """
    Four parameterization modes:
    
    1. 'standard_lr': Standard param with no LR scaling - for finding optimal base LR
       - All initialization as standard but no LR scaling
    
    2. 'mup_no_align_lr': muP param with no LR scaling - for finding optimal base LR
       - All muP initialization and parameter multipliers, but no LR scaling
    
    3. 'standard': Standard param with Adam (no alignment) LR scaling
       - Embedding layer: No scaling with width
       - Hidden layers: Scale by (base_width/width)^0.5
       - Readout layer: Scale by (base_width/width)^0.5
    
    4. 'mup_no_align': muP param with Adam (no alignment) LR scaling
       - Embedding layer: Scale by (base_width/width)^0.5
       - Hidden layers: Scale by (base_width/width)^0.5
       - Readout layer: No scaling with width
    """

    def __init__(
        self,
        d: int,             # input dimension
        hidden_size: int,   # model width n
        depth: int,
        mode: str = 'standard',
        base_width: int = 1024,  # Base width for LR scaling
        embed_lr_scale: float = 1.0,
        hidden_lr_scale: float = 1.0,
        readout_lr_scale: float = 1.0,
        gamma: float = 1.0
    ):
        super().__init__()

        self.mode = mode
        self.depth = depth
        self.hidden_size = hidden_size
        self.input_dim = d
        self.base_width = base_width
        
        # Per-layer LR scales multipliers (user-configurable)
        self.embed_lr_scale = embed_lr_scale
        self.hidden_lr_scale = hidden_lr_scale
        self.readout_lr_scale = readout_lr_scale

        # Factor for forward pass
        self.gamma = gamma

        # We'll store the "relative" LR scale for each Linear in this list
        self.layer_lrs: List[float] = []
        
        layers = []
        prev_dim = d
        
        # Build embedding + hidden layers
        for layer_idx in range(depth):
            linear = nn.Linear(prev_dim, hidden_size)
            
            if mode.startswith('mup_no_align'):
                # muP parameterization (with or without LR scaling)
                if layer_idx == 0:
                    # Embedding layer
                    scale = np.sqrt(hidden_size)  # param multiplier = n^{+1/2}
                    init_std = 1.0 / np.sqrt(hidden_size)
                else:
                    # Hidden layers
                    scale = 1.0
                    init_std = 1.0 / np.sqrt(hidden_size)

                # Apply actual muP parameter initialization
                with torch.no_grad():
                    linear.weight.data = scale * torch.randn(hidden_size, prev_dim) * init_std
                    linear.bias.data.zero_()
                
                # Determine learning rate scaling based on mode
                if mode == 'mup_no_align_lr':
                    # No LR scaling mode - for finding base LR
                    lr_scale = 1.0
                else:  # 'mup_no_align'
                    # Proper theory-based LR scaling for Adam (no alignment)
                    if layer_idx == 0:
                        # Embedding layer: Scale by (base_width/width)^0.5
                        lr_scale = self.embed_lr_scale * (self.base_width / hidden_size) ** 0.5
                    else:
                        # Hidden layer: Scale by (base_width/width)^0.5
                        lr_scale = self.hidden_lr_scale * (self.base_width / hidden_size) ** 0.5
                
                self.layer_lrs.append(lr_scale)

            else:  # 'standard' or 'standard_lr'
                # Standard parameterization
                init_std = 1.0 / np.sqrt(hidden_size)  # stdev ~ 1/sqrt(n)

                # Apply standard parameter initialization
                with torch.no_grad():
                    linear.weight.data = torch.randn(hidden_size, prev_dim) * init_std
                    linear.bias.data.zero_()
                
                # Determine learning rate scaling based on mode
                if mode == 'standard_lr':
                    # No LR scaling mode - for finding base LR
                    lr_scale = 1.0
                else:  # 'standard'
                    # Proper theory-based LR scaling for Adam (no alignment)
                    if layer_idx == 0:
                        # Embedding layer: No scaling with width
                        lr_scale = self.embed_lr_scale
                    else:
                        # Hidden layer: Scale by (base_width/width)^0.5
                        lr_scale = self.hidden_lr_scale * (self.base_width / hidden_size) ** 0.5
                
                self.layer_lrs.append(lr_scale)

            layers.extend([linear, nn.ReLU()])
            prev_dim = hidden_size
        
        # Build readout layer
        final_layer = nn.Linear(prev_dim, 1)
        
        if mode.startswith('mup_no_align'):
            # muP readout
            scale = 1.0 / np.sqrt(hidden_size)  # param multiplier = n^{-1/2}
            init_std = 1.0 / np.sqrt(hidden_size)

            with torch.no_grad():
                final_layer.weight.data = scale * torch.randn(1, hidden_size) * init_std
                final_layer.bias.data.zero_()
            
            # Determine learning rate scaling
            if mode == 'mup_no_align_lr':
                # No LR scaling mode - for finding base LR
                lr_scale = 1.0
            else:  # 'mup_no_align'
                # Proper theory-based LR scaling for Adam (no alignment)
                # Readout layer: No scaling with width
                lr_scale = self.readout_lr_scale
            
            self.layer_lrs.append(lr_scale)
        
        else:  # 'standard' or 'standard_lr'
            # STANDARD readout
            init_std = 1.0 / np.sqrt(hidden_size)

            with torch.no_grad():
                final_layer.weight.data = torch.randn(1, hidden_size) * init_std
                final_layer.bias.data.zero_()
            
            # Determine learning rate scaling
            if mode == 'standard_lr':
                # No LR scaling mode - for finding base LR
                lr_scale = 1.0
            else:  # 'standard'
                # Proper theory-based LR scaling for Adam (no alignment)
                # Readout layer: Scale by (base_width/width)^0.5
                lr_scale = self.readout_lr_scale * (self.base_width / hidden_size) ** 0.5
            
            self.layer_lrs.append(lr_scale)
        
        layers.append(final_layer)
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass, dividing by self.gamma at the end.
        """
        return self.network(x).squeeze() / self.gamma

    def get_layer_learning_rates(self, base_lr: float) -> List[float]:
        """
        Returns the per-layer effective LR = base_lr * (relative scale in self.layer_lrs).
        """
        return [base_lr * lr for lr in self.layer_lrs]