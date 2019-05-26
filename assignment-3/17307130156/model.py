import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RecurrentNetwork(nn.Module):

    def __init__(self, vocab_size, target_size, embedding_dim, hidden_size, num_layers=1, batch_first=False, mode='LSTM'):
        super(RecurrentNetwork, self).__init__()


        self.batch_first = batch_first
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.target_size = target_size # In this case target size is equal to vocab_size


        self.embed = nn.Embedding(vocab_size, embedding_dim)
        # self.embed = nn.Linear(vocab_size, embedding_dim)

        self.rnn = nn.LSTM(embedding_dim,
                           hidden_size,
                           num_layers=num_layers,
                           bias=True,
                           bidirectional=False,
                           # dropout=0.7,
                           batch_first=batch_first)
        self.linear = nn.Linear(hidden_size, target_size)

        self.softmax = nn.LogSoftmax(dim=2)


    def init_hidden(self, batch_size):
        hidden = (Variable(torch.zeros((self.num_layers, batch_size, self.hidden_size))), Variable(torch.zeros((self.num_layers, batch_size, self.hidden_size))))
        self.hidden = hidden
        return hidden


    def forward(self, padded_input, input_lengths):
        # input: (seq_len, batch_size)

        seq_len, batch_size = padded_input.size()
        # print ('seq_len: {}, batch_size: {}'.format(seq_len, batch_size))

        padded_embed = self.embed(padded_input)
        # shape: (seq_len, batch_size, embedding_dim)
        # print ('embed: ', padded_embed.shape)

        packed_in = pack_padded_sequence(padded_embed,
                                         input_lengths,
                                         batch_first=self.batch_first)
        
        packed_out, self.hidden = self.rnn(packed_in, self.hidden)

        padded_out, _ = pad_packed_sequence(packed_out,
                                         batch_first=self.batch_first,
                                         total_length=seq_len)
        # shape: (seq_len, batch_size, hidden_size)
        # print ('unpacked: ', padded_out.shape)

        linear_out = self.linear(padded_out)
        # shape: (seq_len, batch_size, vocab_size)

        output = self.softmax(linear_out)
        # shape: (seq_len, batch_size, vocab_size)

        return output, self.hidden

    
    def loss(self, output, target, padding_value):
        # output shape: (seq_len, batch_size, vocab_size)
        # target shape: (seq_len, batch_size)
        
        output = output.view(-1, self.vocab_size) # (seq_len * batch_size, vocab_size)
        tokens = target.view(-1) # (seq_len * batch_size)

        # != is broadcastable
        mask = (tokens != padding_value).float()

        # Ignoring the padding value, we get the number of total chars of this batch
        num_valid_tokens = int(torch.sum(mask))

        # Pick out softmaxed value for target word and masked out padding
        picked_values = output[range(output.shape[0]), tokens] * mask
        # This is tensor indexing, not slicing, which is output[:, tokens]

        cross_entropy_loss = - torch.sum(picked_values) / num_valid_tokens
        return cross_entropy_loss
        
        
    def perplexity(self, output, target, padding_value):
        return torch.exp(self.loss(output, target, padding_value))
        


