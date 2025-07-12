class RNN(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.3):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, 
                            dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

    def forward(self, x, hidden):
        x = x.long()
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        out = lstm_out.contiguous().view(-1, self.hidden_dim)
        out = self.dropout(out)
        out = self.fc(out)
        return out.view(x.size(0), -1, self.fc.out_features)[:, -1], hidden

    def init_hidden(self, batch_size, device):
        weight = next(self.parameters()).data
        return (weight.new_zeros(self.n_layers, batch_size, self.hidden_dim).to(device),
                weight.new_zeros(self.n_layers, batch_size, self.hidden_dim).to(device))
