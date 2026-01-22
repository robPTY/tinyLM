import torch

import config
from model import TinyLM
from tokenizer import Tokenizer


def make_src_mask(src_ids: torch.Tensor) -> torch.Tensor:
    '''Create a mask for the source sequence.'''
    return (src_ids != tokenizer.PAD).unsqueeze(1).unsqueeze(2)  # src_ids: (1, T)


def make_tgt_mask(tgt_ids: torch.Tensor) -> torch.Tensor:
    '''Create a mask for the target sequence.'''
    tgt_len = tgt_ids.size(1)  # tgt_ids: (1, T)
    pad_mask = (tgt_ids != tokenizer.PAD).unsqueeze(1).unsqueeze(2)  # (1,1,1,T)
    causal = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt_ids.device, dtype=torch.bool))
    return pad_mask & causal  # (1,1,T,T)


@torch.no_grad()
def translate(text: str, max_len: int = 60) -> str:
    '''Return the translated text using TinyLM'''
    model.eval()

    src_ids = tokenizer.encode(text)
    src = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)
    src_mask = make_src_mask(src)

    enc = model.encode(src, src_mask)

    # start with BOS
    generated = torch.tensor([[tokenizer.BOS]], dtype=torch.long, device=device)

    for _ in range(max_len):
        tgt_mask = make_tgt_mask(generated)
        dec = model.decode(generated, src_mask, tgt_mask, enc)
        logits = model.linear_projection(dec)  # (1, T, V)
        next_id = torch.argmax(logits[:, -1, :], dim=-1)  # (1,)
        generated = torch.cat([generated, next_id.unsqueeze(0)], dim=1)

        if next_id.item() == tokenizer.EOS:
            break

    return tokenizer.decode(generated.squeeze(0).tolist())


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = Tokenizer(vocab_size=config.TOTAL_VOCAB_SIZE)
    tokenizer.load("tokenizer_weights.json")

    model = TinyLM(config=config.TinyLM_Config).to(device)

    # Load last checkpoint (or best one so far)
    ckpt = torch.load("weights/tinyLM_bs16_lr1e-4_layers6.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    while True:
        text = input("EN> ").strip()
        if not text:
            break
        print("ES>", translate(text))
