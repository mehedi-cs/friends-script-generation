import torch
import torch.nn.functional as F

def forward_back_prop(rnn, optimizer, criterion, inp, target, hidden, device):
    inp, target = inp.to(device), target.to(device)
    hidden = tuple([h.data for h in hidden])
    rnn.zero_grad()
    out, hidden = rnn(inp, hidden)
    loss = criterion(out, target)
    loss.backward()
    nn.utils.clip_grad_norm_(rnn.parameters(), 5)
    optimizer.step()
    return loss.item(), hidden

def train_rnn(rnn, train_loader, optimizer, criterion, n_epochs, batch_size, device):
    rnn.train()
    for epoch in range(1, n_epochs + 1):
        hidden = rnn.init_hidden(batch_size, device)
        total_loss = 0
        for batch_i, (x, y) in enumerate(train_loader):
            loss, hidden = forward_back_prop(rnn, optimizer, criterion, x, y, hidden, device)
            total_loss += loss
            if (batch_i + 1) % 100 == 0:
                print(f'Epoch {epoch}/{n_epochs} | Batch {batch_i + 1} | Loss: {total_loss / 100:.4f}')
                total_loss = 0
    return rnn
