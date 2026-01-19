import torch
import math
from torch import nn
from typing import Dict
from torch import Tensor
from embedding import Embedding

class MultiHeadAttention(nn.Module): 
    def __init__(self, config: Dict):
        super().__init__()
        self.cfg = config
        self.num_heads = config["N_HEADS"]
        self.d_model = config["D_MODEL"]
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        self.dk = config["D_MODEL"] // config["N_HEADS"] # 512/8 = 64
        self.Wq = nn.Linear(config["D_MODEL"], config["D_MODEL"])
        self.Wk = nn.Linear(config["D_MODEL"], config["D_MODEL"])
        self.Wv = nn.Linear(config["D_MODEL"], config["D_MODEL"])
        self.Wo = nn.Linear(config["D_MODEL"], config["D_MODEL"])
        self.dropout = config["DROPOUT"]
    
    def forward(self, x: Tensor) -> Tensor:
        batch, num_tokens, d_emb = x.shape # (B, T, D)

        # Initial projection
        queries = self.Wq(x) # (16, T, 512) x (512, 512) -> (16, T, 512)
        keys = self.Wk(x) # (16, T, 512) x (512, 512)
        values = self.Wv(x) # (16, T, 512) x (512, 512)

        # Reshape for n heads (B, H, T, D//H)
        queries = queries.view(batch, self.num_heads, num_tokens, self.dk) 
        keys = keys.view(batch, self.num_heads, num_tokens, self.dk) 
        values = values.view(batch, self.num_heads, num_tokens, self.dk)

        # Dot prod of queries and keys
        attn_scores = queries @ keys.transpose(2, 3) # (B, H, T, 64) x (B, H, 64, T)
        scaled_scores = attn_scores / math.sqrt(self.dk)
        attn_weights = torch.softmax(scaled_scores, dim=-1)

        # Dot prod of "weights" and values
        attention = attn_weights @ values # (B, H, T, 64) x (B, H, T, 64)

        attn_out = self.Wo(attention)
        return attn_out

class LayerNorm(nn.Module): 
    def __init__(self, eps: float):
        super().__init__()
        self.eps = eps # For numerical stability and to avoid division by zero
        # nn.Parameter makes them learnable
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplies the xj
        self.bias = nn.Parameter(torch.zeros(1)) # Gets added to xj

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim=True)
        return self.alpha * (x - mean) / torch.sqrt(std + self.eps) + self.bias

class ResidualLayer(nn.Module):
    def __init__(self, config: Dict):
        super().__init__() 
        self.cfg = config
        self.dropout = nn.Dropout(config["DROPOUT"])
        self.norm = LayerNorm() 

    def forward(self, x: Tensor, prev_layer) -> Tensor:
        return x + self.dropout(self.norm(prev_layer(x)))
    
class FeedForwardBlock(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.cfg = config
        self.W1 = nn.Linear(config["D_MODEL"], config["D_FF"]) # Includes the bias b1
        self.W2 = nn.Linear(config["D_FF"], config["D_MODEL"]) # Likewise for b2
        self.dropout = nn.Dropout(config["DROPOUT"])

    def forward(self, x: Tensor) -> Tensor:
        # shape of x: (B, T, D)
        xforward = self.W1(x) # (B, T, D) x (D, DFF) -> (B, T, DFF)
        relud = torch.relu(xforward)
        return self.W2(self.dropout(relud)) # (B, T, DFF) x (DFF, D) -> (B, T, D)

class TinyGPT:
    def __init__(self, config: Dict, device: str):
        self.cfg = config
        self.d_model = config["D_MODEL"]
        self.embedding = Embedding(config["EMBEDDING_SIZE"], self.d_model, config["DROPOUT"], device=device)
    
    def forward(self, X: Tensor) -> Tensor:
        xEmbeddings = self.embedding.forward(X)