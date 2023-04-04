import dill as pickle
from tqdm import tqdm
import numpy as np
import random
import os
import spacy
import codecs
import spacy
import tarfile
import torch
import copy

import torchtext
import re
import string

import torchtext.data
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import random

__all__ = ["prepare_dataloaders", ] 


PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'

src_lang_model = spacy.load('de_core_news_sm')
trg_lang_model = spacy.load('en_core_web_sm')

def prepare_dataloaders(max_len, min_freq, batch_size, device):
    
    # build dataset
    train, val, test = torchtext.datasets.Multi30k(
        root='.wmt16_data', split=('train', 'valid', 'test'), language_pair=('de', 'en'))
    
    # tokenlizer
    train_datasets = TextDatasets(train)
    valid_datasets = TextDatasets(val)
    test_datasets = TextDatasets(test)
    train_datasets_len = len(train_datasets)
    val_datasets_len = len(valid_datasets)
    
    
    # show datasets results after tokenlizer
    print("train_datasets len: ", len(train_datasets))
    print("valid_datasets len: ", len(valid_datasets))
    print("test_datasets len: ", len(test_datasets))
    # for i in range(10):
    #     print(f"++++++++++++++++++++++++++++ train example {i} ++++++++++++++++++++++++++++++++++++")
    #     print("[", len(train_datasets[i][0]), "]", train_datasets[i][0])
    #     print("[", len(train_datasets[i][1]), "]", train_datasets[i][1])
    # print("---------------------------------------------------------------------------------------")
    
    # build src and trg vocabularies
    src_corpus = []
    trg_corpus = []
    for src, trg in train_datasets:
        src_corpus.append(src)
        trg_corpus.append(trg)
    src_vocab = torchtext.vocab.build_vocab_from_iterator(src_corpus, min_freq=min_freq, specials=(PAD_WORD, UNK_WORD, BOS_WORD, EOS_WORD))
    trg_vocab = torchtext.vocab.build_vocab_from_iterator(trg_corpus, min_freq=min_freq, specials=(PAD_WORD, UNK_WORD, BOS_WORD, EOS_WORD))
    src_vocab.set_default_index(0)
    trg_vocab.set_default_index(0)

    print('[Info] Get source language vocabulary size:', len(src_vocab.vocab))
    # print("src_vocab vars: ")
    # print(vars(src_vocab).keys())
    # print("src_vocab.vocab.itos[:20]: ")
    # print(src_vocab.vocab.get_itos()[:20])
    # print("src_vocab.vocab.stoi[:20]: ")
    # print(list(src_vocab.vocab.get_stoi().items())[:20])
    
    print('[Info] Get target language vocabulary size:', len(trg_vocab.vocab))
    # print("trg_vocab.vocab vars: ")
    # print(vars(trg_vocab).keys())
    # print("trg_vocab.vocab.itos[:20]: ")
    # print(trg_vocab.vocab.get_itos()[:20])
    # print("trg_vocab.vocab.stoi[:20]: ")
    # print(list(trg_vocab.vocab.get_stoi().items())[:20])
    
    # merge two vocabularies
    print('[Info] Merging two vocabulary ...')
    src_vocab_stoi = src_vocab.get_stoi()
    trg_vocab_stoi = trg_vocab.get_stoi()
    trg_vocab_itos = trg_vocab.get_itos()
    for w, _ in src_vocab_stoi.items():
        # TODO: Also update the `freq`, although it is not likely to be used.
        if w not in trg_vocab_stoi:
            trg_vocab_stoi[w] = len(trg_vocab_stoi)
    trg_vocab_itos = [None] * len(trg_vocab_stoi)
    for w, i in trg_vocab_stoi.items():
        trg_vocab_itos[i] = w
    src_vocab_stoi = trg_vocab_stoi
    src_vocab_itos = trg_vocab_itos
    print('[Info] Get merged vocabulary size:', len(src_vocab_stoi))
    
    # convert datasets samples from strings to ints by vocabularies
    # if x is not in src_vocab_stoi, set it to default_value
    def src_transform(x, default_value=src_vocab_stoi[PAD_WORD]):
        result = []
        for token in x:
            if token in src_vocab_stoi.keys():
                result.append(src_vocab_stoi[token])
            else:
                result.append(default_value)  
        return result
    
    def trg_transform(x, default_value=src_vocab_stoi[PAD_WORD]):
        result = []
        for token in x:
            if token in trg_vocab_stoi.keys():
                result.append(trg_vocab_stoi[token])
            else:
                result.append(default_value)
        return result
    
    # transform src_seq and trg_seq using vocabulary, and padding sequences to the same length in a batch
    def collate_batch(batch):
        output_src_list = []
        output_trg_list = []
        for (_src, _trg) in batch:
            processed_src = torch.tensor(src_transform(_src))
            processed_trg = torch.tensor(trg_transform(_trg))
            output_src_list.append(processed_src)
            output_trg_list.append(processed_trg)
        return {"src": pad_sequence(output_src_list, padding_value=src_vocab_stoi[PAD_WORD]).to(device),
                "trg": pad_sequence(output_trg_list, padding_value=trg_vocab_stoi[PAD_WORD]).to(device)}
    
    # sort sequences to make sequences of same length be together, reducing padding
    def train_batch_sampler(src_train_list, batch_size):
        indices = [(i, len(s[0])) for i, s in enumerate(src_train_list)]
        random.shuffle(indices)
        pooled_indices = []
        # create pool of indices with similar lengths 
        for i in range(0, len(indices), batch_size * 100):
            pooled_indices.extend(sorted(indices[i:i + batch_size * 100], key=lambda x: x[1]))
        pooled_indices = [x[0] for x in pooled_indices]
        
        # yield indices for infinite batches
        while True:
            for i in range(0, len(pooled_indices), batch_size):
                yield pooled_indices[i:i + batch_size]
            
    def val_batch_sampler(trg_train_list, batch_size):
        indices = [(i, len(s[0])) for i, s in enumerate(trg_train_list)]
        random.shuffle(indices)
        pooled_indices = []
        # create pool of indices with similar lengths 
        for i in range(0, len(indices), batch_size * 100):
            pooled_indices.extend(sorted(indices[i:i + batch_size * 100], key=lambda x: x[1]))
        pooled_indices = [x[0] for x in pooled_indices]
        
        # yield indices for infinite batches
        while True:
            for i in range(0, len(pooled_indices), batch_size):
                yield pooled_indices[i:i + batch_size]

    # build dataloader
    train_iterator = DataLoader(list(train_datasets), batch_sampler=train_batch_sampler(train_datasets, batch_size), collate_fn=collate_batch)
    val_iterator = DataLoader(list(valid_datasets), batch_sampler=val_batch_sampler(valid_datasets, batch_size), collate_fn=collate_batch)
    
    # prepare other output parameters
    src_pad_idx = src_vocab_stoi[PAD_WORD]
    trg_pad_idx = trg_vocab_stoi[PAD_WORD]
    trg_bos_idx = trg_vocab_stoi[BOS_WORD]
    trg_eos_idx = trg_vocab_stoi[EOS_WORD]
    unk_idx = src_vocab_stoi[UNK_WORD]
    src_vocab_size = len(src_vocab_stoi)
    trg_vocab_size = len(src_vocab_stoi)

    return train_iterator, val_iterator, test_datasets, train_datasets_len, val_datasets_len, src_pad_idx, trg_pad_idx, \
        trg_bos_idx, trg_eos_idx, src_vocab_size, trg_vocab_size, unk_idx, \
            src_vocab_stoi, trg_vocab_itos, BOS_WORD, EOS_WORD  

def preprocessing_text(text):
    text = text.lower().strip()
    # text = re.sub(f'[{string.punctuation}\n]', '', text)
    return text

def tokenize_de(text):
    text = preprocessing_text(text)
    return [tok.text for tok in src_lang_model.tokenizer(text)]

def tokenize_en(text):
    text = preprocessing_text(text)
    return [tok.text for tok in trg_lang_model.tokenizer(text)]
                      
class TextDatasets(torch.utils.data.Dataset):
    def __init__(self, raw_data):
        self.datasets = list(raw_data)
        
    def __len__(self):
        return len(self.datasets)
    
    def __getitem__(self, idx):
        src, trg = self.datasets[idx]
        src = [BOS_WORD] + tokenize_de(src) + [EOS_WORD]
        trg = [BOS_WORD] + tokenize_en(trg) + [EOS_WORD]
        return src, trg
