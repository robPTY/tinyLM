from typing import Dict
from embedding import Embedding
from torch import Tensor

class MultiHeadAttention: 
    pass 

class TransformerBlock: 
    def __init__(self):
        pass 

    def forward(self, X: Tensor):
        pass

class TinyGPT:
    def __init__(self, config: Dict, device: str):
        self.cfg = config
        self.embedding = Embedding(self.cfg.EMBEDDING_SIZE, self.cfg.D_MODEL, device=device)
    
    def forward(self, X: Tensor) -> Tensor:
        xEmbeddings = self.embedding.forward(X)