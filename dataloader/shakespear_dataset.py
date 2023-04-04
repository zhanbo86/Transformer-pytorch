"""define dataset"""
import os
import sys
import torch


class ShakespeareDataset():
    def __init__(self) -> None:
        with open('./dataloader/input.txt', 'r', encoding='utf-8') as f:
            text = f.read()

        # list chars in text
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        print(''.join(chars))
        print(self.vocab_size)

        # create a map from characters to ints
        stoi = {ch:i for i,ch in enumerate(chars)}
        itos = {i:ch for i,ch in enumerate(chars)}
        self.encode = lambda s: [stoi[c] for c in s]
        self.decode = lambda l: ''.join([itos[i] for i in l])

        # encode the whole dataset
        data = torch.tensor(self.encode(text), dtype=torch.long)
        print(data.shape, data.dtype)
        # print(data[:1000])

        # split up the data into train and test
        n = int(0.9*len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]
        
    def get_batch(self, split, batch_size, block_size):
        data = self.train_data if split=="train" else self.val_data
        ix = torch.randint(len(data)-block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        return x, y
