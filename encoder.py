from typing import Dict

from torch import Tensor, nn

from layers import FeedForwardBlock, LayerNorm, MultiHeadAttention, ResidualLayer


class EncoderBlock(nn.Module):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.config = config
        self.attention_block = MultiHeadAttention(config)
        self.feed_forward = FeedForwardBlock(config)
        # Each block has 2 residual connections
        self.residuals = nn.ModuleList([ResidualLayer(config) for _ in range(2)])

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        # The first residual will need to call Norm(x + Dropout(Attention(x, mask)))
        x = self.residuals[0](x, lambda x: self.attention_block(x, x, x, mask))
        # The second residual will call Norm(x + Dropout(FFN(x)))
        x = self.residuals[1](x, self.feed_forward)
        return x


class Encoder(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.encoder_layers = nn.ModuleList([EncoderBlock(config) for _ in range(config["N_LAYERS"])])
        self.norm = LayerNorm(config["D_MODEL"], config["EPS"])

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        for layer in self.encoder_layers:
            x = layer(x, mask)
        return self.norm(x)
