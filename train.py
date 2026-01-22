from typing import Callable, Dict, List, Tuple, cast

import pandas as pd
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import config
from model import TinyLM
from tokenizer import Tokenizer
from translation_dataset import TranslationDataset

DatasetSplit = Tuple[pd.Series, pd.Series, pd.Series, pd.Series]
Batch = List[Tuple[List[int], List[int]]]
CollateOutput = Dict[str, Tensor]


def load_dataset(path: str) -> DatasetSplit:
    '''Return the dataset split into train and test sets'''
    df = pd.read_csv(path, sep="\t", header=None, names=["english", "spanish", "meta"])
    df = df[["english", "spanish"]]

    Xs = cast(pd.Series, df["english"])
    Ys = cast(pd.Series, df["spanish"])

    # 90, 10 split
    dataset_size = Xs.shape[0]
    train_size = int(dataset_size * 0.90)

    X_train = cast(pd.Series, Xs[:train_size])
    Y_train = cast(pd.Series, Ys[:train_size])
    X_test = cast(pd.Series, Xs[train_size:])
    Y_test = cast(pd.Series, Ys[train_size:])

    return X_train, Y_train, X_test, Y_test


def make_collate_fn(tokenizer: Tokenizer, max_len: int | None) -> Callable[[Batch], CollateOutput]:
    '''Return a collate function that pads sequences to max_len'''
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

def forward_step(model: TinyLM, batch: CollateOutput, criterion: nn.Module,
    device: torch.device, pad_id: int) -> Tuple[Tensor, int]:
    '''Returns the loss for a single batch's forward pass of the model.'''
    xInputs = batch["src_input_ids"].to(device) # (B, T) for encoder
    yInputs = batch["tgt_input_ids"].to(device) # (B, T) for decoder
    xMask = batch["src_attention_mask"].to(device) # (B, T) for encoder
    yMask = batch["tgt_attention_mask"].to(device) # (B, T) for decoder

    # Slice the inputs for decoder (feed everything except last token)
    yInputs_in = yInputs[:, :-1]
    yMask_in = yMask[:, :-1]

    # Expand masks for attention
    src_mask = xMask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)

    target_length = yInputs_in.size(1)
    causal_mask = torch.tril(torch.ones((target_length, target_length), device=device, dtype=torch.bool))
    target_padding_mask = yMask_in.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T-1)
    target_mask = target_padding_mask & causal_mask  # (B, 1, T-1, T-1)

    encoder_out = model.encode(xInputs, src_mask) # (B, T, D)
    decoder_out = model.decode(yInputs_in, src_mask, target_mask, encoder_out) # (B, T, D)
    logits = model.linear_projection(decoder_out) # (B, T-1, V)

    labels = yInputs[:, 1:].contiguous().view(-1) # (B, T) -> (B*T)
    loss = criterion(logits.reshape(-1, logits.size(-1)), labels) # (B*T, V)

    num_tokens = int((labels != pad_id).sum().item())

    return loss, num_tokens

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Training on device: {device}...')

    # Load dataset
    xTrain, yTrain, xTest, yTest = load_dataset(config.FILE_PATH)

    # Load tokenizer
    tokenizer = Tokenizer(vocab_size=config.TOTAL_VOCAB_SIZE)
    tokenizer.load("tokenizer_weights.json")

    # Pass the tokenized data from Dataset to Dataloader
    training_set = TranslationDataset(xTrain, yTrain, tokenizer)
    test_set = TranslationDataset(xTest, yTest, tokenizer)
    train_loader = DataLoader(
        training_set,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=make_collate_fn(tokenizer, config.MAX_SEQ_LEN),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=make_collate_fn(tokenizer, config.MAX_SEQ_LEN),
    )

    # Initialize model
    model = TinyLM(config=config.TinyLM_Config).to(device)

    # Initialize TensorBoard writer
    writer = SummaryWriter()

    # Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, betas=(config.BETA1, config.BETA2), eps=config.EPS)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.PAD, label_smoothing=config.E_LS).to(device)

    global_step = 0
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0.0
        total_tokens = 0

        for batch in train_loader:
            optimizer.zero_grad()  # Clear gradients
            loss, num_tokens = forward_step(model, batch, criterion, device, tokenizer.PAD)

            loss.backward()  # Backprop the loss and update weights
            optimizer.step()

            total_loss += loss.item() * num_tokens # Update loss and log to TensorBoard
            total_tokens += num_tokens

            writer.add_scalar("Loss/train_batch", loss.item(), global_step)
            global_step += 1

            if global_step % 10 == 0:
                print(f'Step {global_step} | Loss: {loss.item():.4f}')

        train_epoch_loss = total_loss / total_tokens
        writer.add_scalar("Loss/train_epoch", train_epoch_loss, epoch)
        print(f'Training loss (token-avg): {train_epoch_loss}')

        # Get validation loss
        model.eval()
        validation_loss = 0.0
        validation_tokens = 0

        with torch.no_grad():
            for batch in test_loader:
                loss, num_tokens = forward_step(model, batch, criterion, device, tokenizer.PAD)
                validation_loss += loss.item() * num_tokens
                validation_tokens += num_tokens

        val_epoch_loss = validation_loss / validation_tokens
        writer.add_scalar("Loss/validation_epoch", val_epoch_loss, epoch)
        print(f'Validation loss (token-avg): {val_epoch_loss}')

        # Save model at each epoch
        model_filename = f"{config.MODEL_BASENAME}_{epoch}.pt"
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step
            }, model_filename)

if __name__ == "__main__":
    main()
