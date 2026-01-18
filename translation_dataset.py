import torch
import pandas as pd 
from tokenizer import Tokenizer 
from typing import Tuple 
from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    def __init__(self, X: pd.Series, Y: pd.Series, tokenizer: Tokenizer):
        self.X = X.reset_index(drop=True)
        self.Y = Y.reset_index(drop=True)
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int) -> Tuple[str, str]:
        xTokens = self.tokenizer.encode(self.X.iloc[index])
        yTokens = self.tokenizer.encode(self.Y.iloc[index])
        return xTokens, yTokens