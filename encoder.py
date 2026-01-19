from torch import nn
from typing import Dict
from torch import Tensor
from model import LayerNorm, MultiHeadAttention, FeedForwardBlock, ResidualLayer

class EncoderBlock(nn.Module):
    def __init__(self, config: Dict):
        super().__init__() 
        self.config = config
        self.attention_block = MultiHeadAttention
        self.feed_forward = FeedForwardBlock
        # Each block has 2 residual connections
        self.residuals = nn.ModuleList([ResidualLayer(config["DROPOUT"]) for _ in range(2)])
    
    def forward(self):
        pass

class Encoder(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.encoder_layers = [EncoderBlock() for _ in range(config["N_LAYERS"])]
        self.norm = LayerNorm() 

    def forward(self, x: Tensor, mask):
        for layer in self.encoder_layers:
            x = layer(x, mask)
        return self.norm(x)