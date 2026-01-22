from typing import List, Tuple

import pandas as pd
from torch.utils.data import Dataset

from tokenizer import Tokenizer


class TranslationDataset(Dataset):
    def __init__(self, X: pd.Series, Y: pd.Series, tokenizer: Tokenizer) -> None:
        self.X = X.reset_index(drop=True)
        self.Y = Y.reset_index(drop=True)
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int) -> Tuple[List[int], List[int]]:
        '''Return a tuple of tokenized input and output sequences'''
        xTokens = self.tokenizer.encode(self.X.iloc[index])
        yTokens = self.tokenizer.encode(self.Y.iloc[index])
        return xTokens, yTokens
