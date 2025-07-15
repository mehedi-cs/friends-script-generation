import numpy as np
import torch
import torch.nn.functional as F

def generate(rnn, prime_id, int_to_vocab, token_dict, pad_value, predict_len, sequence_length, device):
    rnn.eval()
    current_seq = np.full((1, sequence_length), pad_value)
    current_seq[-1][-1] = prime_id
    predicted = [int_to_vocab[prime_id]]
    
    for _ in range(predict_len):
        seq = torch.LongTensor(current_seq).to(device)
        hidden = rnn.init_hidden(1, device)
        output, _ = rnn(seq, hidden)
        p = F.softmax(output, dim=1).data.cpu().numpy().squeeze()
        top_i = np.argsort(p)[-5:]
        word_i = np.random.choice(top_i, p=p[top_i]/p[top_i].sum())
        predicted.append(int_to_vocab[word_i])
        current_seq = np.roll(current_seq, -1, 1)
        current_seq[0, -1] = word_i
    
    result = ' '.join(predicted)
    for key, token in token_dict.items():
        result = result.replace(f' {token.lower()}', key)
    return result
