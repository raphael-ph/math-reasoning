# This is a script to hold any commom Modules and structures that can
# be shared around multiple implementations.

from typing import Optional

import torch
import torch.nn as nn

# internal modules
from ..utils.logger import get_logger

_logger = get_logger(__name__, level="DEBUG")

# --- MLP ---
# A Simple Implementation of a Multi Layer Perceptron
class MLP(nn.Module):
    """Implementation of a generic multilayer perceptron"""
    def __init__(self, 
                 in_dim: int,
                 out_dim: int,
                 hidden_dim: int,
                 hidden_layers: int,
                 activation: nn.Module = nn.ReLU(), # defaults to ReLU
                 dropout: Optional[float] = None,
                 ):
        super().__init__()
        self.out_dim = out_dim
        self.dropout = dropout
        dim = in_dim

        # creating the MLP
        layers = []
        for i in range(hidden_layers):
            layers += [
                nn.Linear(dim, hidden_dim),
                activation
            ]
            dim = hidden_dim
        
        # adding the output layer
        layers.append(nn.Linear(dim, out_dim))

        if dropout:
            layers.append(nn.Dropout(dropout))
        
        self.model = nn.Sequential(*layers)
        total_params = sum(param.numel() for param in self.model.parameters())
        _logger.debug(f"Total MLP size: {total_params}")
    
    def forward(self, x):
        y = self.model(x)
        return y
    
if __name__ == "__main__":
    net = MLP(5, 1, 20, 4, 0.2)
    print(net)