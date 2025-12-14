# --- The Transformer Architecture ---
# Inspired by: 
# Karpathy's (as always) video on GPT-2: http://youtube.com/watch?v=kCc8FmEb1nY
# Attention is all you need (of course): https://arxiv.org/abs/1706.03762
# 
# Ths script implements the complete Transformer Architecture, as defined by the 2017 paper;

import torch
import torch.nn as nn
import torch.nn.functional as F

# internal imports
from .common import MLP

# --- Attention Head ---
class AttentionHead(nn.Module):
    """Single head of Attention"""
    def __init__(self, emb_dim: int, head_size: int):
        super().__init__()
        # The authors use a fixed d_model = 512. That is our "head_size"
        # This is a fixed dim throughout the whole attention module. As per the paper:
        # >>> "To facilitate these residual connections, all sub-layers in the model, \
        # >>> as well as the embedding layers, produce outputs of dimension d_model = 512."
        self.k = nn.Linear(emb_dim, head_size, bias=False)
        self.q = nn.Linear(emb_dim, head_size, bias=False)
        self.v = nn.Linear(emb_dim, head_size, bias=False)

    def forward(self, x):
        """Implementation of the forward pass of the Attention Head
        
        This implementation is true to the 'Attention is all you need' paper, following
        the same steps:
        >>> Matmul(Q, K) = weights -> Scale(weights) -> Mask(weights) -> Softmax(weights) = out -> Matmul(out, V)
        """
        B, T, C = x.shape
        # Key, Query first go through a Linear transformation. 
        K = self.k(x) # (B, T, C) -> batch, context window, channels (embeddings)
        Q = self.q(x)

        # Matmul(Q, K) = weights and Scale(weights)
        # following the paper implementation, after the Matmul, the weights
        # are scaled, dividing by sqrt(d_keys).
        wei = Q @ K.transpose(-2, -1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)

        # Masking(weights)
        # Since this implementation is for the decoder, it is necessary to apply the 
        # masking on the "future" information, which means the tokens that are still to come.
        # Karpathy's implementation uses the torch.register_buffer() function.
        # TO DO: Explore which are the pros and cons of my current implementation
        tril = torch.ones(T, T).tril()
        wei = wei.masked_fill(tril == 0, float("-inf"))

        # Softmax(weights)
        out = F.softmax(wei, -1) # apply softmax to the Channel dim, so we get the probs for the embeddings

        # Matmul(weights, V)
        V = self.v(x) # (B, T, C)
        out = out @ V # (B, T, T) @ (B, T, C) -> (B, T, C)

        return out

