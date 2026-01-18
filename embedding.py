import torch
import math 
from torch import Tensor 

class Embedding:
    def __init__(self, vocab_size: int, d_model: int, device: str):
        self.d_model = d_model
        self.device = device
        self.We = torch.randn((vocab_size, d_model), device=self.device) * 0.1 # embedding weights (8k, 512)

    def get_positional_embeddings(self, T: int):
        positions = torch.arange(T, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Generate positions in jumps of 2 (to match the 2i from paper)
        i = torch.arange(0, self.d_model, 2, device=self.device)
        denominator = torch.exp((i / self.d_model) * math.log(10000)) # a^b = e^(b * ln(a))

        pe = torch.zeros((T, self.d_model), device=self.device)
        pe[:, 0::2] = torch.sin(positions/denominator)
        pe[:, 1::2] = torch.cos(positions/denominator)

        return pe

    def forward(self, x: Tensor) -> Tensor:
        # Manuscript multiplies weights by sqrt(d_model)
        embedding = self.We[x] * math.sqrt(self.d_model)

        # Add positional encodings
        embedding = embedding + self.get_positional_embeddings(x.shape[1])
        return embedding