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

# setting up device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

# --- Transformer ---
class Transformer(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, context_size: int, n_layers: int, n_heads):
        super().__init__()
        # variables
        self.context_size = context_size

        self.embeddings = nn.Embedding(vocab_size, emb_dim)
        self.pos_encoding = nn.Embedding(self.context_size, emb_dim)
        self.blocks = nn.Sequential(*[Block(n_heads, emb_dim) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(emb_dim)
        self.linear = nn.Linear(emb_dim, vocab_size)
    
    def forward(self, idx, targets=None):
        """Implementing the forward pass on the Transformer"""
        # the input to the transformer are the tokens, shaped (B, T), where B is the
        # batch size and T the temporal component (context window). As per the paper,
        # tokens are converted into embeddings. 
        B, T = idx.shape
        token_embeddings = self.embeddings(idx) # (B, T) -> (B, T, C) adding the channels (embeddings) dim
        # Generate the positional encoding embeddings. For this, we will simply initialize the embeddings
        # with the size of the temporal component, which will provide us a form of representing the position that the words have in the sentence.
        # The authors explain that since the transformer has no Recurrence nor Convolution, it is necessary to create means for the 
        # model to use the sequence order, hence, the positional encoding:
        # >>> "Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, 
        # >>> we must inject some information about the relative or absolute position of the tokens in the sequence.""
        #
        # In the paper, the authors use a Sine function to represent the positional encoding.
        # TO DO: implement Sine PE
        positional_encoding = self.pos_encoding(torch.arange(T, device=DEVICE)) # (B, T) -> (B, T, C) adds embeddings for the position
        x = token_embeddings + positional_encoding
        x = self.blocks(x)
        x = self.layer_norm(x)
        logits = self.linear(x) # (B, T, C) -> (B, T, vocab_size)
        
        # This logic allows us to use the forward pass for training and for generating:
        # if targets are provided, the model will be trained. If targets are not passed, the model simply outputs
        # the logits, which are then used on softmax.
        if not targets:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(-1)

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """Implementing the inference pass of the transformer"""
        for _ in range(max_new_tokens):
            B, T = idx.shape
            # first, we crop the input tokens and select only the last tokens that fit on context size
            idx_context = idx[:, -self.context_size:]
            # do a forward pass
            logits, loss = self.forward(idx_context) # (B, T, C)
            # focus only on last time step
            logits = logits[:, -1, :] # (B, C)
            # softmax and sampling
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            # now we concatenate the next token on the sequence
            idx = torch.cat((idx, next_token), dim=-1)

        return idx, loss