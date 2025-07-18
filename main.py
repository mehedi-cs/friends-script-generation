from src.data_loader import *
from src.model import RNN
from src.train import train_rnn
from src.generate import generate
import torch
import torch.nn as nn
import os

DATA_DIR = './data/friends_scripts'
SEQ_LEN = 8
BATCH_SIZE = 256
EPOCHS = 25
LR = 0.0003
EMBED_DIM = 256
HIDDEN_DIM = 512
LAYERS = 3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

text = load_scripts(DATA_DIR)
int_to_vocab, vocab_to_int, int_text = create_vocab(text)
train_loader = get_dataloader(int_text, SEQ_LEN, BATCH_SIZE)

vocab_size = len(vocab_to_int)
rnn = RNN(vocab_size, vocab_size, EMBED_DIM, HIDDEN_DIM, LAYERS, dropout=0.5).to(DEVICE)
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

trained_rnn = train_rnn(rnn, train_loader, optimizer, criterion, EPOCHS, BATCH_SIZE, DEVICE)
torch.save(trained_rnn.state_dict(), './models/rnn_trained.pt')

pad_value = vocab_to_int['<PAD>']
prime_word = 'joey'
generated = generate(trained_rnn, vocab_to_int[prime_word], int_to_vocab, pun_dic(), pad_value, 400, SEQ_LEN, DEVICE)
print(generated)
