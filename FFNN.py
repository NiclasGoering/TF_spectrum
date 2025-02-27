import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # Added this import
from typing import List

class DeepNN(nn.Module):
    """
    Parameterization modes:
    
    1. '*_lr' modes: Basic parameterizations without learning rate scaling 
       (for finding optimal base LR)
       - 'standard_lr': Standard parameterization 
       - 'ntk_lr': NTK parameterization
       - 'mup_lr': muP parameterization
    
    2. Regular modes with alignment option:
       - 'standard': Standard parameterization
       - 'ntk': NTK parameterization
       - 'mup': muP parameterization
       
       When alignment=False:
       - Standard LR scaling: Embedding (1), Hidden (1/√n), Readout (1/√n)
       - NTK LR scaling: Embedding (1), Hidden (1), Readout (1)
       - muP LR scaling: Embedding (1/√n), Hidden (1/√n), Readout (1)
       
       When alignment=True:
       - Standard LR scaling: Embedding (1), Hidden (1/n), Readout (1/n)
       - NTK LR scaling: Embedding (1), Hidden (1/√n), Readout (1/√n)
       - muP LR scaling: Embedding (1/√n), Hidden (1/n), Readout (1/√n)
    """

    def __init__(
        self,
        d: int,             # input dimension
        hidden_size: int,   # model width n
        depth: int,
        mode: str = 'standard',
        alignment: bool = False,  # New parameter to control alignment assumption
        base_width: int = 1024,   # Base width for LR scaling
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
        self.alignment = alignment
        
        # Per-layer LR scales multipliers (user-configurable)
        self.embed_lr_scale = embed_lr_scale
        self.hidden_lr_scale = hidden_lr_scale
        self.readout_lr_scale = readout_lr_scale

        # Factor for forward pass
        self.gamma = gamma

        # We'll store the "relative" LR scale for each Linear in this list
        self.layer_lrs: List[float] = []
        
        # Store all layers, parameter multipliers, and configurations
        self.layers = nn.ModuleList()
        self.param_multipliers = []
        
        prev_dim = d
        
        # Build embedding + hidden layers
        for layer_idx in range(depth):
            linear = nn.Linear(prev_dim, hidden_size)
            is_embedding = (layer_idx == 0)
            
            # Step 1: Configure initialization variance and parameter multiplier
            if mode.startswith('standard'):
                # Standard parameterization
                if is_embedding:
                    # Embedding layer - initialization variance ~ 1
                    init_std = 1.0
                    param_multiplier = 1.0  # param multiplier = 1
                else:
                    # Hidden layers - initialization variance ~ 1/n
                    init_std = 1.0 / np.sqrt(hidden_size)
                    param_multiplier = 1.0  # param multiplier = 1
                
            elif mode.startswith('ntk'):
                # NTK parameterization
                if is_embedding:
                    # Embedding layer - initialization variance ~ 1
                    init_std = 1.0
                    param_multiplier = 1.0  # param multiplier = 1
                else:
                    # Hidden layers - initialization variance ~ 1
                    init_std = 1.0
                    param_multiplier = 1.0 / np.sqrt(hidden_size)  # param multiplier = 1/√n
            
            elif mode.startswith('mup'):
                # muP parameterization
                if is_embedding:
                    # Embedding layer - initialization variance ~ 1/n
                    init_std = 1.0 / np.sqrt(hidden_size)
                    param_multiplier = np.sqrt(hidden_size)  # param multiplier = √n
                else:
                    # Hidden layers - initialization variance ~ 1/n
                    init_std = 1.0 / np.sqrt(hidden_size)
                    param_multiplier = 1.0  # param multiplier = 1
            
            # Apply initialization with correct variance
            with torch.no_grad():
                linear.weight.data = torch.randn(hidden_size, prev_dim) * init_std
                linear.bias.data.zero_()
            
            # Step 2: Determine learning rate scaling
            # If it's a *_lr mode, set learning rate scale to 1.0 (no scaling)
            if mode.endswith('_lr'):
                lr_scale = 1.0
            else:
                # Standard and variants
                if mode.startswith('standard'):
                    if is_embedding:
                        # Embedding layer: No scaling with width
                        lr_scale = self.embed_lr_scale
                    else:
                        # Hidden layer scaling depends on alignment
                        if alignment:
                            # Full alignment: Scale by (base_width/width)^1.0
                            lr_scale = self.hidden_lr_scale * (self.base_width / hidden_size) ** 1.0  # LR ~ 1/n
                        else:
                            # No alignment: Scale by (base_width/width)^0.5
                            lr_scale = self.hidden_lr_scale * (self.base_width / hidden_size) ** 0.5  # LR ~ 1/√n
                
                # NTK and variants
                elif mode.startswith('ntk'):
                    if is_embedding:
                        # Embedding layer: No scaling with width
                        lr_scale = self.embed_lr_scale
                    else:
                        # Hidden layer scaling depends on alignment
                        if alignment:
                            # Full alignment: Scale by (base_width/width)^0.5
                            lr_scale = self.hidden_lr_scale * (self.base_width / hidden_size) ** 0.5  # LR ~ 1/√n
                        else:
                            # No alignment: No scaling with width
                            lr_scale = self.hidden_lr_scale  # LR ~ 1
                
                # muP and variants
                elif mode.startswith('mup'):
                    if is_embedding:
                        # Embedding layer: Scale by (base_width/width)^0.5
                        lr_scale = self.embed_lr_scale * (self.base_width / hidden_size) ** 0.5  # LR ~ 1/√n
                    else:
                        # Hidden layer scaling depends on alignment
                        if alignment:
                            # Full alignment: Scale by (base_width/width)^1.0
                            lr_scale = self.hidden_lr_scale * (self.base_width / hidden_size) ** 1.0  # LR ~ 1/n
                        else:
                            # No alignment: Scale by (base_width/width)^0.5
                            lr_scale = self.hidden_lr_scale * (self.base_width / hidden_size) ** 0.5  # LR ~ 1/√n
            
            self.layer_lrs.append(lr_scale)
            self.param_multipliers.append(param_multiplier)
            self.layers.append(linear)
            self.layers.append(nn.ReLU())
            prev_dim = hidden_size
        
        # Build readout layer
        final_layer = nn.Linear(prev_dim, 1)
        
        # Configure readout layer based on mode
        if mode.startswith('standard'):
            # Standard readout - initialization variance ~ 1/n
            init_std = 1.0 / np.sqrt(hidden_size)
            param_multiplier = 1.0  # param multiplier = 1
            
        elif mode.startswith('ntk'):
            # NTK readout - initialization variance ~ 1
            init_std = 1.0
            param_multiplier = 1.0 / np.sqrt(hidden_size)  # param multiplier = 1/√n
            
        elif mode.startswith('mup'):
            # muP readout - initialization variance ~ 1/n
            init_std = 1.0 / np.sqrt(hidden_size)
            param_multiplier = 1.0 / np.sqrt(hidden_size)  # param multiplier = 1/√n
        
        # Apply initialization with correct variance
        with torch.no_grad():
            final_layer.weight.data = torch.randn(1, hidden_size) * init_std
            final_layer.bias.data.zero_()
        
        # Determine learning rate scaling for readout layer
        if mode.endswith('_lr'):
            lr_scale = 1.0
        else:
            # Standard and variants
            if mode.startswith('standard'):
                if alignment:
                    # Full alignment: Scale by (base_width/width)^1.0
                    lr_scale = self.readout_lr_scale * (self.base_width / hidden_size) ** 1.0  # LR ~ 1/n
                else:
                    # No alignment: Scale by (base_width/width)^0.5
                    lr_scale = self.readout_lr_scale * (self.base_width / hidden_size) ** 0.5  # LR ~ 1/√n
            
            # NTK and variants
            elif mode.startswith('ntk'):
                if alignment:
                    # Full alignment: Scale by (base_width/width)^0.5
                    lr_scale = self.readout_lr_scale * (self.base_width / hidden_size) ** 0.5  # LR ~ 1/√n
                else:
                    # No alignment: No scaling with width
                    lr_scale = self.readout_lr_scale  # LR ~ 1
            
            # muP and variants
            elif mode.startswith('mup'):
                if alignment:
                    # Full alignment: Scale by (base_width/width)^0.5
                    lr_scale = self.readout_lr_scale * (self.base_width / hidden_size) ** 0.5  # LR ~ 1/√n
                else:
                    # No alignment: No scaling with width
                    lr_scale = self.readout_lr_scale  # LR ~ 1
        
        self.layer_lrs.append(lr_scale)
        self.param_multipliers.append(param_multiplier)
        self.layers.append(final_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with parameter multipliers applied correctly.
        """
        linear_idx = 0
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                # Calculate weight multiplication
                weight_output = F.linear(x, layer.weight)
                
                # Apply parameter multiplier to the result of weight multiplication
                weight_output = weight_output * self.param_multipliers[linear_idx]
                
                # Add bias (which isn't shown in the paper's formulation)
                if layer.bias is not None:
                    x = weight_output + layer.bias
                else:
                    x = weight_output
                    
                linear_idx += 1
            else:
                # For non-linear layers (ReLU), just apply normally
                x = layer(x)
        
        # Return result divided by gamma
        return x.squeeze() / self.gamma

    def get_layer_learning_rates(self, base_lr: float) -> List[float]:
        """
        Returns the per-layer effective LR = base_lr * (relative scale in self.layer_lrs).
        """
        return [base_lr * lr for lr in self.layer_lrs]