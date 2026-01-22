import math
from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor

class MultiHeadAttention(nn.Module):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.cfg = config
        self.num_heads = config["N_HEADS"]
        self.d_model = config["D_MODEL"]
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        self.dk = config["D_MODEL"] // config["N_HEADS"]  # 512/8 = 64
        self.Wq = nn.Linear(config["D_MODEL"], config["D_MODEL"], bias=config["QKV_BIAS"])
        self.Wk = nn.Linear(config["D_MODEL"], config["D_MODEL"], bias=config["QKV_BIAS"])
        self.Wv = nn.Linear(config["D_MODEL"], config["D_MODEL"], bias=config["QKV_BIAS"])
        self.Wo = nn.Linear(config["D_MODEL"], config["D_MODEL"], bias=config["QKV_BIAS"])
        self.dropout = nn.Dropout(config["DROPOUT"])

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor) -> Tensor:
        '''Project inputs into multiple heads, and compute attention scores with optional masking.'''
        batch, q_len, d_emb = q.shape  # (B, T, D)
        k_len = k.shape[1]
        v_len = v.shape[1]

        # Initial projection
        queries = self.Wq(q)  # (B, T, 512) x (512, 512) -> (B, T, 512)
        keys = self.Wk(k)  # (B, T, 512) x (512, 512) -> (B, T, 512)
        values = self.Wv(v)  # (B, T, 512) x (512, 512) -> (B, T, 512)

        # Reshape for n heads (B, T, H, D//H) -> (B, H, T, D//H)
        queries = queries.view(batch, q_len, self.num_heads, self.dk).transpose(1, 2)
        keys = keys.view(batch, k_len, self.num_heads, self.dk).transpose(1, 2)
        values = values.view(batch, v_len, self.num_heads, self.dk).transpose(1, 2)

        # Dot prod of queries and keys
        attn_scores = queries @ keys.transpose(2, 3)  # (B, H, T, 64) x (B, H, 64, T)
        scaled_scores = attn_scores / math.sqrt(self.dk)

        # Apply masking before softmax
        if mask is not None:
            scaled_scores = scaled_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(scaled_scores, dim=-1) # (B, H, T, T)

        # Dot prod of "weights" and values
        attention = attn_weights @ values  # (B, H, T, 64) x (B, H, T, 64)

        # (B, H, T, D//H) -> (B, T, H, D//H) -> (B, T, D)
        attention = attention.transpose(1, 2).contiguous().view(batch, q_len, d_emb)

        attn_out = self.Wo(attention) # (B, T, D) x (D, D)
        return attn_out


class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float) -> None:
        super().__init__()
        self.eps = eps  # For numerical stability and to avoid division by zero
        # nn.Parameter makes them learnable
        self.alpha = nn.Parameter(torch.ones(d_model))  # Multiplies the xj
        self.bias = nn.Parameter(torch.zeros(d_model))  # Gets added to xj

    def forward(self, x: Tensor) -> Tensor:
        '''Normalize input tensor using mean and variance, then scale + shift with learnable parameters.'''
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.std(dim=-1, unbiased=False, keepdim=True)
        return self.alpha * (x - mean) / torch.sqrt(variance + self.eps) + self.bias


class ResidualLayer(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.cfg = config
        self.dropout = nn.Dropout(config["DROPOUT"])
        self.norm = LayerNorm(config["D_MODEL"], config["EPS"])

    def forward(self, x: Tensor, sublayer: nn.Module) -> Tensor:
        '''Execute the Pre-Norm residual connection logic: Norm(x + Dropout(Sublayer(x))).'''
        return self.norm(x + self.dropout(sublayer(x)))

class LinearLayer(nn.Module):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.cfg = config
        # So, in the manuscript they mention in section 3.4 that the pre-softmax transformation weights
        # are the same ones used in the embedding layers
        self.projection = nn.Linear(config["D_MODEL"], config["EMBEDDING_SIZE"])

    def forward(self, x: Tensor) -> Tensor:
        '''Perform a linear transformation to project the input tensor to the output embedding space.'''
        return self.projection(x) # (B, T, D) x (D, V) -> (B, T, V)

class FeedForwardBlock(nn.Module):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.cfg = config
        self.W1 = nn.Linear(config["D_MODEL"], config["D_FF"])  # Includes the bias b1
        self.W2 = nn.Linear(config["D_FF"], config["D_MODEL"])  # Likewise for b2
        self.dropout = nn.Dropout(config["DROPOUT"])

    def forward(self, x: Tensor) -> Tensor:
        '''Processes inputs through a d_model -> d_ff -> d_model transformation with a non-linear activation.'''
        # shape of x: (B, T, D)
        xforward = self.W1(x)  # (B, T, D) x (D, DFF) -> (B, T, DFF)
        relud = torch.relu(xforward)
        return self.W2(self.dropout(relud))  # (B, T, DFF) x (DFF, D) -> (B, T, D)
