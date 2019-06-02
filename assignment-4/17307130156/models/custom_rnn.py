import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class RNNText(nn.Module):

    def __init__(self, 
                 vocab_size,
                 output_size,
                 embedding_dim=256,
                 hidden_size=128,
                 num_layers=1,
                 bidirectional=True,
                 batch_first=False,
                 dropout=0.5,
                 mode='LSTM'):

        super(RNNText, self).__init__()

        self.batch_first = batch_first
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional == True else 1

        self.embed = nn.Embedding(vocab_size, embedding_dim)

        self.rnn = nn.LSTM(embedding_dim,
                           hidden_size,
                           num_layers=num_layers,
                           bias=True,
                           bidirectional=bidirectional,
                           # dropout=0.7,
                           batch_first=False)

        self.linear = nn.Linear(hidden_size * self.num_directions, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, padded_input, input_lengths):

        if self.batch_first:
            padded_input = passed_input.permute(1, 0)
        # (seq_len, batch_size)

        seq_len, batch_size = padded_input.size()

        padded_embed = self.embed(padded_input)
        # shape: (seq_len, batch_size, embedding_dim)

        packed_in = pack_padded_sequence(padded_embed,
                                         input_lengths,
                                         batch_first=False)
        
        packed_out, (hidden, cell) = self.rnn(packed_in)

        padded_out, _ = pad_packed_sequence(packed_out,
                                            batch_first=self.batch_first,
                                            total_length=seq_len)

        hidden = hidden.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)

        # Extract last hidden state of last layer 
        hidden = torch.cat((hidden[-1, -2, :, :], hidden[-1, -1, :, :]), dim=1)
        hidden = self.dropout(hidden)
        # shape: (batch_size, hidden_size * num_directions)

        output = self.linear(hidden)
        # shape: (batch_size, output_size)
        
        return output
        # return {'pred': output}



if __name__ == '__main__':


    sentence1 = [1, 2, 3]
    sentence2 = [1, 2, 3, 1]
    sentences = [sentence1, sentence2]

    def indexed_seqs_to_input(seqs):
        # seqs: [(seq1_len), ...]
        seqs = [torch.LongTensor(seq) for seq in seqs]
        seqs.sort(key=lambda x: x.shape[0], reverse=True)

        sorted_lengths = [seq.shape[0] for seq in seqs]
        padded_input = pad_sequence(seqs)
        # (seq_len, batch_size)

        return padded_input, sorted_lengths
        
    padded_input, sorted_lengths = indexed_seqs_to_input(sentences)
    model = RNNText(vocab_size=4, output_size=5)
    print (model(padded_input, sorted_lengths).shape)
    



