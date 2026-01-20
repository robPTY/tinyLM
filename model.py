import math
from typing import Dict

import torch
from torch import Tensor, nn
from torch.types import Device

from embedding import Embedding
from encoder import Encoder


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

    def forward(self, x: Tensor, mask) -> Tensor:
        batch, num_tokens, d_emb = x.shape  # (B, T, D)

        # Initial projection
        queries = self.Wq(x)  # (B, T, 512) x (512, 512) -> (B, T, 512)
        keys = self.Wk(x)  # (B, T, 512) x (512, 512) -> (B, T, 512)
        values = self.Wv(x)  # (B, T, 512) x (512, 512) -> (B, T, 512)

        # Reshape for n heads (B, T, H, D//H) -> (B, H, T, D//H)
        queries = queries.view(batch, num_tokens, self.num_heads, self.dk).transpose(1, 2)
        keys = keys.view(batch, num_tokens, self.num_heads, self.dk).transpose(1, 2)
        values = values.view(batch, num_tokens, self.num_heads, self.dk).transpose(1, 2)

        # Dot prod of queries and keys
        attn_scores = queries @ keys.transpose(2, 3)  # (B, H, T, 64) x (B, H, 64, T)
        scaled_scores = attn_scores / math.sqrt(self.dk)

        # Apply masking before softmax
        if mask:
            scaled_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(scaled_scores, dim=-1) # (B, H, T, T)

        # Dot prod of "weights" and values
        attention = attn_weights @ values  # (B, H, T, 64) x (B, H, T, 64)

        # (B, H, T, D//H) -> (B, T, H, D//H) -> (B, T, D)
        attention = attention.transpose(1, 2).contiguous().view(batch, num_tokens, d_emb)

        attn_out = self.Wo(attention) # (B, T, D) x (D, D)
        return attn_out


class LayerNorm(nn.Module):
    def __init__(self, eps: float) -> None:
        super().__init__()
        self.eps = eps  # For numerical stability and to avoid division by zero
        # nn.Parameter makes them learnable
        self.alpha = nn.Parameter(torch.ones(1))  # Multiplies the xj
        self.bias = nn.Parameter(torch.zeros(1))  # Gets added to xj

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / torch.sqrt(std + self.eps) + self.bias


class ResidualLayer(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.cfg = config
        self.dropout = nn.Dropout(config["DROPOUT"])
        self.norm = LayerNorm(config["EPS"])

    def forward(self, x: Tensor, sublayer: nn.Module) -> Tensor:
        return self.norm(x + self.dropout(sublayer(x)))


class FeedForwardBlock(nn.Module):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.cfg = config
        self.W1 = nn.Linear(config["D_MODEL"], config["D_FF"])  # Includes the bias b1
        self.W2 = nn.Linear(config["D_FF"], config["D_MODEL"])  # Likewise for b2
        self.dropout = nn.Dropout(config["DROPOUT"])

    def forward(self, x: Tensor) -> Tensor:
        # shape of x: (B, T, D)
        xforward = self.W1(x)  # (B, T, D) x (D, DFF) -> (B, T, DFF)
        relud = torch.relu(xforward)
        return self.W2(self.dropout(relud))  # (B, T, DFF) x (DFF, D) -> (B, T, D)


class TinyGPT:
    def __init__(self, config: Dict, device: Device):
        self.cfg = config
        self.d_model = config["D_MODEL"]
        self.embedding = Embedding(config["EMBEDDING_SIZE"], self.d_model, config["DROPOUT"], device=device)
        self.encoder = Encoder(config)

    def forward(self, X: Tensor) -> Tensor:
        xEmbeddings = self.embedding.forward(X)
        return xEmbeddings
