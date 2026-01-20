import math

import torch
from torch import Tensor, nn
from torch.types import Device


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, dropout: float, device: Device):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.device = device
        self.We = nn.Embedding(vocab_size, d_model)  # embedding weights (8k, 512)
        self.dropout = nn.Dropout(dropout)  # For positional encodings

    def get_positional_embeddings(self, T: int) -> Tensor:
        # (T, 1)
        positions = torch.arange(T, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Generate positions in jumps of 2 (to match the 2i from paper)
        i = torch.arange(0, self.d_model, 2, device=self.device)
        denominator = torch.exp((i / self.d_model) * math.log(10000))  # a^b = e^(b * ln(a))

        pe = torch.zeros((T, self.d_model), device=self.device)  # (T, 512)
        # Sin for evens, cos for odds
        pe[:, 0::2] = torch.sin(positions / denominator)
        pe[:, 1::2] = torch.cos(positions / denominator)

        return pe  # (T, 512)

    def forward(self, x: Tensor) -> Tensor:
        # Manuscript multiplies weights by sqrt(d_model)
        embedding = self.We(x) * math.sqrt(self.d_model)

        # Add positional encodings (in manuscript they are fixed)
        embedding = embedding + self.get_positional_embeddings(x.shape[1]).requires_grad_(False)  # (B, T, D)
        return self.dropout(embedding)
