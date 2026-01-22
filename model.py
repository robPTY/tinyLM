from typing import Dict

from torch import Tensor, nn

from embedding import Embedding
from encoder import Encoder
from decoder import Decoder
from layers import LinearLayer

class TinyLM(nn.Module):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.cfg = config
        self.d_model = config["D_MODEL"]
        self.embedding = Embedding(config["EMBEDDING_SIZE"], self.d_model, config["DROPOUT"])
        self.linear_layer = LinearLayer(config)
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def encode(self, x: Tensor, source_mask: Tensor) -> Tensor:
        '''Perform a forward pass through the Transformer encoder, returning the encoder output.'''
        xEmbeddings = self.embedding.forward(x)
        encoder_out = self.encoder.forward(xEmbeddings, source_mask)
        return encoder_out

    def decode(self, y: Tensor, source_mask: Tensor, target_mask: Tensor, encoder_out: Tensor) -> Tensor:
        '''Perform a forward pass through the Transformer decoder, returning the decoder output.'''
        yEmbeddings = self.embedding.forward(y)
        decoder_out = self.decoder.forward(yEmbeddings, source_mask, target_mask, encoder_out)
        return decoder_out

    def linear_projection(self, x: Tensor) -> Tensor:
        '''Project the decoder's output to the vocabulary dimension to produce logits.'''
        return self.linear_layer.forward(x)
