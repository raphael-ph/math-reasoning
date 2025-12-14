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
    def __init__(self, emb_dim: int, head_size: int, dropout: float = 0.2):
        super().__init__()
        # The authors use a fixed d_model = 512. That is our "head_size"
        # This is a fixed dim throughout the whole attention module. As per the paper:
        # >>> "To facilitate these residual connections, all sub-layers in the model, \
        # >>> as well as the embedding layers, produce outputs of dimension d_model = 512."
        self.k = nn.Linear(emb_dim, head_size, bias=False)
        self.q = nn.Linear(emb_dim, head_size, bias=False)
        self.v = nn.Linear(emb_dim, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Implementation of the forward pass of the Attention Head
        
        This implementation is true to the 'Attention is all you need' paper, following
        the same steps:
        >>> Matmul(Q, K) = weights -> Scale(weights) -> Mask(weights) -> Softmax(weights) -> Matmul(wei, V)
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
        wei = F.softmax(wei, -1) # apply softmax to the Channel dim, so we get the probs for the embeddings

        # Adding a dropout after computing the Softmax;
        # This simulates that we destroy some tracks of communication, forcing the network to learn
        # more robust representations.
        wei = self.dropout(wei)

        # Matmul(weights, V)
        V = self.v(x) # (B, T, C)
        out = wei @ V # (B, T, T) @ (B, T, C) -> (B, T, C)

        return out

# --- Multi-Head Attention ---
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads: int, emb_dim: int, head_size: int, dropout: float = 0.2):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(emb_dim, head_size) for _ in range(n_heads)])
        self.linear = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Applying the Multi-Head Attention in parallel
        
        The idea is to run the attention mechanism on the input in parallel. With this implementation,
        the net will learn different aspects of the input in each layer. As the authors state:

        >>> "Multi-head attention allows the model to jointly attend to information from different representation \ 
        >>> subspaces at different positions. With a single attention head, averaging inhibits this."
        """
        out = torch.cat([h(x) for h in self.heads], dim=-1) # concatenate on the last dim, Channels
        # applying the linear projection, as stated by the authors
        out = self.linear(out)
        out = self.dropout(out)

        return out

# --- Transformer Block ---
class Block(nn.Module):
    def __init__(self, n_heads: int, emb_dim: int):
        """Implementation of the Transformer block. 
        
        Inspired by Karpathy, this implementation does not have the "second" multi-head attention
        presented in the original architecture. This only implements the Masked Attention.
        """
        super().__init__()
        head_size = emb_dim//n_heads

        # "core" of the transformer
        self.self_attention = MultiHeadAttention(n_heads, emb_dim, head_size)
        self.ffwd = MLP(in_dim=emb_dim, out_dim=emb_dim, hidden_dim=4*emb_dim, hidden_layers=1) # simpe ffwd net

        # normalization
        self.layer_norm1 = nn.LayerNorm(emb_dim)
        self.layer_norm2 = nn.LayerNorm(emb_dim)
    
    def forward(self, x):
        # adding the skip connections
        x = x + self.self_attention(self.layer_norm1(x))
        x = x + self.ffwd(self.layer_norm2(x))

        return x
