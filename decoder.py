from torch import nn, Tensor

from layers import FeedForwardBlock, LayerNorm, MultiHeadAttention, ResidualLayer

class DecoderBlock(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.cfg = config
        self.masked_attention = MultiHeadAttention(config)
        self.unmasked_attention = MultiHeadAttention(config)
        self.feed_forward = FeedForwardBlock(config)
        self.residuals = nn.ModuleList([ResidualLayer(config) for _ in range(3)])

    def forward(self, x: Tensor, source_mask: Tensor, target_mask: Tensor, encoder_output: Tensor) -> Tensor:
        '''Process input through masked self-attention, encoder-decoder cross-attention, and feed-forward sub-layers with residual connections.'''
        # First residual will be of Norm(x + Dropout(Masked_MHA(x, mask)))
        x = self.residuals[0](x, lambda x: self.masked_attention(x, x, x, target_mask))
        # Second residual will be of Norm(x + Dropout(MHA(x, mask))) with encoder output as q and k
        x = self.residuals[1](x, lambda x: self.unmasked_attention(x, encoder_output, encoder_output, source_mask))
        # Third residual will be of Norm(x + Dropout(FFN(x)))
        x = self.residuals[2](x, self.feed_forward)
        return x

class Decoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.decoder_layers = nn.ModuleList([DecoderBlock(config) for _ in range(config["N_LAYERS"])])
        self.norm = LayerNorm(config["D_MODEL"], config["EPS"])

    def forward(self, x: Tensor, source_mask: Tensor, target_mask: Tensor, encoder_output: Tensor) -> Tensor:
        '''Passes the input through the sequence of decoder blocks.'''
        for layer in self.decoder_layers:
            x = layer(x, source_mask, target_mask, encoder_output)
        return self.norm(x)
