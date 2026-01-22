import json
import pandas as pd
from typing import List, Tuple, Dict

class Tokenizer:
    def __init__(self, vocab_size: int) -> None:
        self.byte_length = 256
        self.vocab_size = vocab_size
        self.BOS = self.vocab_size
        self.EOS = self.vocab_size + 1
        self.PAD = self.vocab_size + 2
        self.tokens = []
        self.merges = {}

    def get_vocab(self) -> Dict[int, bytes]:
        '''Return a dictionary mapping token IDs to their corresponding bytes.'''
        vocab = {i: bytes([i]) for i in range(256)}
        if len(self.merges) > 0:
            for (p0, p1), index in self.merges.items():
                vocab[index] = vocab[p0] + vocab[p1]
        return vocab

    def tokenize(self, X: pd.Series) -> List[List[int]]:
        '''Train BPE tokenizer by merging most frequent pairs.'''
        assert self.vocab_size >= 256
        merges = {}
        merge_count = self.vocab_size - self.byte_length
        tokens = self.bpe(X)
        for i in range(merge_count):
            pairs = self.get_pairs(tokens)
            if not pairs:
              break
            top_pair = max(pairs.items(), key=lambda x: (x[1], x[0]))[0]
            next_id = i + 256
            tokens = self.merge(tokens, top_pair, next_id)
            merges[top_pair] = i + self.byte_length

        self.merges = merges
        self.tokens = tokens
        return tokens

    def bpe(self, X: pd.Series) -> List[List[int]]:
        '''Tokenize a series of sentences using Byte Pair Encoding.'''
        tokens = []
        for sentence in X:
            x = [self.BOS] + list(map(int, sentence.encode('utf-8'))) + [self.EOS]
            tokens.append(x) # each sentence is <BOS>sentence.<EOS>
        return tokens

    def get_pairs(self, tokens: List[List[int]]) -> Dict[Tuple, int]:
        '''Return a dictionary of token pairs and their frequencies.'''
        pairs = {}
        for x in tokens:
            content = x[1:-1]
            for i in range(len(content)-1):
                token_pair = (content[i], content[i+1])
                pairs[token_pair] = 1 + pairs.get(token_pair, 0)
        return pairs

    def merge(self, tokens: List[List[int]], pair: Tuple[int, int], new_id: int) -> List[List[int]]:
        '''Return a new corpus with merged token pairs given a pair and new ID.'''
        new_corpus = []
        for sentence in tokens:
          new_sentence = []
          index = 0
          while index < len(sentence):
              if index < len(sentence) - 1 and sentence[index] == pair[0] and sentence[index+1] == pair[1]:
                  new_sentence.append(new_id)
                  index += 2
              else:
                  new_sentence.append(sentence[index])
                  index += 1
          new_corpus.append(new_sentence)
        return new_corpus

    def encode(self, text: str) -> List[int]:
        '''Encode a string into a list of token IDs.'''
        if not text:
            return []

        tokens = [self.BOS] + list(text.encode('utf-8')) + [self.EOS]
        corpus = [tokens]
        for pair, new_index in self.merges.items():
            corpus = self.merge(corpus, pair, new_index)

        return corpus[0]

    def decode(self, tokens: List[int]) -> str:
        '''Decode a list of token IDs into a string.'''
        vocab = self.get_vocab()
        to_decode = b"".join(vocab.get(token, b'') for token in tokens if token < self.vocab_size)
        text = to_decode.decode('utf-8', errors="replace")
        return text

    def save(self, file_path: str) -> None:
        '''Save the tokenizer and merges to a file.'''
        merges_list = []

        # Create a list of all merges
        for (x0, x1), new_id in self.merges.items():
            merges_list.append([x0, x1, new_id])

        data = {
            "vocab_size": self.vocab_size,
            "byte_length": self.byte_length,
            "BOS": self.BOS,
            "EOS": self.EOS,
            "PAD": self.PAD,
            "merges": merges_list,
        }

        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, file_path: str) -> None:
        '''Load the tokenizer (mainly the merges) from a file.'''
        with open(file_path, 'r') as f:
            data = json.load(f)

        self.vocab_size = data["vocab_size"]
        self.byte_length = data["byte_length"]
        self.BOS = data["BOS"]
        self.EOS = data["EOS"]
        self.PAD = data["PAD"]

        self.merges = {}
        for x0, x1, new_id in data["merges"]:
            self.merges[(x0, x1)] = new_id
