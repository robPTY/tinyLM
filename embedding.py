import math

import torch
from torch import Tensor, nn


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.We = nn.Embedding(vocab_size, d_model)  # embedding weights (8k, 512)
        self.dropout = nn.Dropout(dropout)  # For positional encodings

    def get_positional_embeddings(self, T: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        '''Create positional embeddings for the given sequence length.'''
        # (T, 1)
        positions = torch.arange(T, device=device, dtype=dtype).unsqueeze(1)

        # Generate positions in jumps of 2 (to match the 2i from paper)
        i = torch.arange(0, self.d_model, 2, device=device, dtype=dtype)
        denominator = torch.exp((i / self.d_model) * math.log(10000))  # a^b = e^(b * ln(a))

        pe = torch.zeros((T, self.d_model), device=device, dtype=dtype) # (T, 512)
        # Sin for evens, cos for odds
        pe[:, 0::2] = torch.sin(positions / denominator)
        pe[:, 1::2] = torch.cos(positions / denominator)

        return pe  # (T, 512)

    def forward(self, x: Tensor) -> Tensor:
        '''Embed the input tokens and add positional encodings.'''
        # Manuscript multiplies weights by sqrt(d_model)
        embedding = self.We(x) * math.sqrt(self.d_model)

        # Add positional encodings (in manuscript they are fixed)
        pe = self.get_positional_embeddings(x.shape[1], x.device, embedding.dtype).requires_grad_(False)
        embedding = embedding + pe
        return self.dropout(embedding)
