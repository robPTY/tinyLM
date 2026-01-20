from typing import Callable, Dict, List, Tuple, cast

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader

import config
from model import TinyGPT
from tokenizer import Tokenizer
from translation_dataset import TranslationDataset

DatasetSplit = Tuple[pd.Series, pd.Series, pd.Series, pd.Series]
Batch = List[Tuple[List[int], List[int]]]
CollateOutput = Dict[str, Tensor]


def load_dataset(path: str) -> DatasetSplit:
    df = pd.read_csv(path, sep="\t", header=None, names=["english", "spanish", "meta"])
    df = df[["english", "spanish"]]

    Xs = cast(pd.Series, df["english"])
    Ys = cast(pd.Series, df["spanish"])

    # 80, 20 split
    vocab_size = Xs.shape[0]
    train_size = int(vocab_size * 0.80)

    X_train = cast(pd.Series, Xs[:train_size])
    Y_train = cast(pd.Series, Ys[:train_size])
    X_test = cast(pd.Series, Xs[train_size:])
    Y_test = cast(pd.Series, Ys[train_size:])

    return X_train, Y_train, X_test, Y_test


def make_collate_fn(
    tokenizer: Tokenizer, max_len: int | None
) -> Callable[[Batch], CollateOutput]:
    pad_id = tokenizer.PAD

    def collate(batch):
        # batch is already a list of (xTokens, yTokens) tuples
        xIds, yIds = zip(*batch)

        if max_len is not None:
            xIds = [x[:max_len] for x in xIds]
            yIds = [x[:max_len] for x in yIds]

        xMax = max(len(x) for x in xIds)
        yMax = max(len(x) for x in yIds)

        def pad_to(xs, T):
            # Create a matrix filled with pad tokens
            out = torch.full((len(xs), T), pad_id, dtype=torch.long)
            # Create a mask of values to ignore
            mask = torch.zeros((len(xs), T), dtype=torch.bool)
            # For each sequence in batch, replace first len(x) with real tokens
            for i, x in enumerate(xs):
                x = torch.tensor(x, dtype=torch.long)
                out[i, : len(x)] = x
                # Mark the first len(x) as true for attention
                mask[i, : len(x)] = True
            return out, mask

        # Tokens, masks for English
        src_input_ids, src_attn_mask = pad_to(xIds, xMax)
        # Tokens, masks for Spanish
        tgt_input_ids, tgt_attn_mask = pad_to(yIds, yMax)

        return {
            "src_input_ids": src_input_ids,
            "src_attention_mask": src_attn_mask,
            "tgt_input_ids": tgt_input_ids,
            "tgt_attention_mask": tgt_attn_mask,
        }

    return collate


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xTrain, yTrain, xTest, yTest = load_dataset(config.FILE_PATH)

    # Tokenize data
    tokenizer = Tokenizer(vocab_size=config.VOCAB_SIZE)
    tokenizer.load("tokenizer_weights.json")

    # Pass tokenized data to Dataset, Dataloader
    training_set = TranslationDataset(xTrain, yTrain, tokenizer)
    test_set = TranslationDataset(xTest, yTest, tokenizer)
    train_loader = DataLoader(
        training_set,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=make_collate_fn(tokenizer, config.MAX_SEQ_LEN),
    )

    # Initialize model
    model = TinyGPT(config=config.TinyGPT_Config, device=device)

    # Batch the data before passing into embedding layer
    for batch in train_loader:
        xInputs = batch["src_input_ids"].to(device)
        yInputs = batch["tgt_input_ids"].to(device)
        break


if __name__ == "__main__":
    main()
