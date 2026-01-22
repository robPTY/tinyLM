import argparse
from itertools import islice

import torch
import evaluate

import config
from model import TinyLM
from tokenizer import Tokenizer
from train import load_dataset


def make_src_mask(src_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    '''Build a padding mask for encoder inputs'''
    return (src_ids != pad_id).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    '''Build a padding mask for the decoder inputs'''
    tgt_len = tgt_ids.size(1)
    pad_mask = (tgt_ids != pad_id).unsqueeze(1).unsqueeze(2)
    causal = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt_ids.device, dtype=torch.bool))
    return pad_mask & causal


@torch.no_grad()
def translate(text: str, max_len: int) -> str:
    '''Turn one English sentence into Spanish'''
    model.eval()

    src_ids = tokenizer.encode(text)
    src = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)
    src_mask = make_src_mask(src, tokenizer.PAD)

    enc = model.encode(src, src_mask)

    generated = torch.tensor([[tokenizer.BOS]], dtype=torch.long, device=device)
    for _ in range(max_len):
        tgt_mask = make_tgt_mask(generated, tokenizer.PAD)
        dec = model.decode(generated, src_mask, tgt_mask, enc)
        logits = model.linear_projection(dec)
        next_id = torch.argmax(logits[:, -1, :], dim=-1)
        generated = torch.cat([generated, next_id.unsqueeze(0)], dim=1)
        if next_id.item() == tokenizer.EOS:
            break

    return tokenizer.decode(generated.squeeze(0).tolist())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint .pt")
    parser.add_argument("--max_len", type=int, default=60)
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric = evaluate.load("sacrebleu")

    tokenizer = Tokenizer(vocab_size=config.TOTAL_VOCAB_SIZE)
    tokenizer.load("tokenizer_weights.json")

    model = TinyLM(config=config.TinyLM_Config).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    _, _, xTest, yTest = load_dataset(config.FILE_PATH)

    refs = []
    hypotheses = []

    for src, ref in islice(zip(xTest, yTest), args.limit):
        hyp = translate(src, max_len=args.max_len)
        refs.append(ref)
        hypotheses.append(hyp)

    references = [[r] for r in refs]
    result = metric.compute(predictions=hypotheses, references=references)
    if result is None:
        raise RuntimeError("metric.compute returned None, check your inputs")

    print(f"BLEU-4: {result['score']:.2f}")
