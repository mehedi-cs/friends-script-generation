import glob
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def load_scripts(data_dir):
    text = ""
    for file in glob.glob(data_dir + "/*.txt"):
        with open(file, 'r') as f:
            text += f.read()
    return text

def create_vocab(text, special_words={'PADDING': '<PAD>'}):
    from collections import Counter
    text = text.lower()
    for key, token in pun_dic().items():
        text = text.replace(key, f' {token} ')
    text = text.split()
    text += list(special_words.values())
    freq_word = Counter(text)
    vocab_sorted = sorted(freq_word, key=freq_word.get, reverse=True)
    int_to_vocab = {i: word for i, word in enumerate(vocab_sorted)}
    vocab_to_int = {word: i for i, word in int_to_vocab.items()}
    int_text = [vocab_to_int[word] for word in text]
    return int_to_vocab, vocab_to_int, int_text

def get_dataloader(text, seq_length, batch_size):
    n_batches = len(text) // batch_size
    text = text[:n_batches * batch_size]
    features, targets = [], []
    for i in range(len(text) - seq_length):
        features.append(text[i:i + seq_length])
        targets.append(text[i + seq_length])
    data = TensorDataset(torch.tensor(features), torch.tensor(targets))
    return DataLoader(data, batch_size=batch_size, shuffle=True)

def pun_dic():
    return {
        '.': '||period||', ',': '||comma||', '"': '||quotation_mark||',
        ';': '||semicolon||', '!': '||exclamation_mark||', '?': '||question_mark||',
        '(': '||left_parentheses||', ')': '||right_parentheses||',
        '-': '||dash||', '\n': '||return||'
    }
