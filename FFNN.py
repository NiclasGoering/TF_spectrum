import numpy as np
import torch
import torch.nn as nn
from typing import List

class DeepNN(nn.Module):
    """
    Two parameterization modes under Adam (no alignment), now with a gamma factor 
    and using 1/sqrt(n) for all layers in 'standard'.

    Mode 1) 'standard' with 1/sqrt(n):
        - Embedding / Hidden / Readout layers:
          * Init stdev = 1 / sqrt(n)  (instead of variance=1 for embedding)
          * Param multiplier = 1
          * LR scale = (layer_lr_scale / sqrt(n)) by default (exposed as embed_lr_scale, 
            hidden_lr_scale, readout_lr_scale). If you want purely 1/sqrt(n), you 
            can set those to 1.0, so that the final scale is 1/sqrt(n).
        - Forward pass divides the output by gamma.

    Mode 2) 'mup_no_align' (same as before):
        - Embedding layer (layer_idx=0):
          * Param multiplier = n^{+1/2}
          * Init stdev = n^{-1}     (variance = 1/n => stdev=1/sqrt(n))
          * LR scale = 1/sqrt(n)
        - Hidden layers (1 <= layer_idx < depth):
          * Param multiplier = 1
          * Init stdev = n^{-1}  
          * LR scale = 1/sqrt(n)
        - Readout layer (final):
          * Param multiplier = n^{-1/2}
          * Init stdev = n^{-1}
          * LR scale = O(1)
        - Forward pass also divides by gamma, for consistency.

    The user can override the default per-layer LR scaling factors (embed_lr_scale, etc.)
    to achieve different overall scalings.
    """

    def __init__(
        self,
        d: int,             # input dimension
        hidden_size: int,   # model width n
        depth: int,
        mode: str = 'standard',
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
        
        # Per-layer LR scales for "standard" (by default we do 1/sqrt(n))
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
            
            if mode == 'mup_no_align':
                # muP + Adam (no alignment), same logic as your previous code
                if layer_idx == 0:
                    # Embedding layer
                    scale = np.sqrt(hidden_size)  # param multiplier = n^{+1/2}
                    init_std = 1.0 / hidden_size  # stdev=1/sqrt(n) from var=1/n
                    lr_scale = 1.0 / np.sqrt(hidden_size)
                else:
                    # Hidden layers
                    scale = 1.0
                    init_std = 1.0 / hidden_size
                    lr_scale = 1.0 / np.sqrt(hidden_size)

                with torch.no_grad():
                    linear.weight.data = scale * torch.randn(hidden_size, prev_dim) * init_std
                    linear.bias.data.zero_()
                
                self.layer_lrs.append(lr_scale)

            else:
                # ===== STANDARD param with 1/sqrt(n) for all layers =====
                init_std = 1.0 / np.sqrt(hidden_size)  # stdev ~ 1/sqrt(n)

                if layer_idx == 0:
                    # Embedding layer 
                    # final LR scale => embed_lr_scale / sqrt(n) if embed_lr_scale=1 => 1/sqrt(n).
                    lr_scale = self.embed_lr_scale / np.sqrt(hidden_size)
                else:
                    # Hidden layers
                    lr_scale = self.hidden_lr_scale / np.sqrt(hidden_size)

                with torch.no_grad():
                    linear.weight.data = torch.randn(hidden_size, prev_dim) * init_std
                    linear.bias.data.zero_()
                
                self.layer_lrs.append(lr_scale)

            layers.extend([linear, nn.ReLU()])
            prev_dim = hidden_size
        
        # Build readout layer
        final_layer = nn.Linear(prev_dim, 1)
        
        if mode == 'mup_no_align':
            # muP readout
            scale = 1.0 / np.sqrt(hidden_size)  # param multiplier = n^{-1/2}
            init_std = 1.0 / hidden_size        # stdev = 1/sqrt(n)
            lr_scale = 1.0  # O(1)

            with torch.no_grad():
                final_layer.weight.data = scale * torch.randn(1, hidden_size) * init_std
                final_layer.bias.data.zero_()
            
            self.layer_lrs.append(lr_scale)
        
        else:
            # STANDARD readout: also stdev=1/sqrt(n)
            init_std = 1.0 / np.sqrt(hidden_size)
            lr_scale = self.readout_lr_scale / np.sqrt(hidden_size)

            with torch.no_grad():
                final_layer.weight.data = torch.randn(1, hidden_size) * init_std
                final_layer.bias.data.zero_()
            
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
        E.g. for standard param with hidden_size=1024, if hidden_lr_scale=1.0 then 
        the hidden layers get ~ base_lr / sqrt(1024).
        """
        return [base_lr * lr for lr in self.layer_lrs]
