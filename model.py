from typing import Dict

from torch import Tensor, nn

from embedding import Embedding
from encoder import Encoder
from decoder import Decoder
from layers import LinearLayer

class TinyGPT(nn.Module):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.cfg = config
        self.d_model = config["D_MODEL"]
        self.embedding = Embedding(config["EMBEDDING_SIZE"], self.d_model, config["DROPOUT"])
        self.linear_layer = LinearLayer(config)
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def encode(self, x: Tensor, source_mask: Tensor) -> Tensor:
        xEmbeddings = self.embedding.forward(x)
        encoder_out = self.encoder.forward(xEmbeddings, source_mask)
        return encoder_out

    def decode(self, y: Tensor, source_mask: Tensor, target_mask: Tensor, encoder_out: Tensor) -> Tensor:
        yEmbeddings = self.embedding.forward(y)
        decoder_out = self.decoder.forward(yEmbeddings, source_mask, target_mask, encoder_out)
        return decoder_out

    def linear_projection(self, x: Tensor) -> Tensor:
        return self.linear_layer.forward(x)
