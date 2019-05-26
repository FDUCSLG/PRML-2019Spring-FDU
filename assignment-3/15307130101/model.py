import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


class LSTMcell(nn.Module):
    def __init__(self, input_size , hidden_size):
        super(LSTMcell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_f = nn.Linear(self.input_size + self.hidden_size, self.hidden_size)
        self.W_i = nn.Linear(self.input_size + self.hidden_size, self.hidden_size)
        self.W_C = nn.Linear(self.input_size + self.hidden_size, self.hidden_size)
        self.W_o = nn.Linear(self.input_size + self.hidden_size, self.hidden_size)

    def forward(self, input, hx=None):
        if not hx:
            pre_state = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
            pre_cell_state = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        else:
            pre_state = hx[0]
            pre_cell_state = hx[1]
        x = torch.cat((input, pre_state), -1)
        ft = torch.sigmoid(self.W_f(x))
        it = torch.sigmoid(self.W_i(x))
        ct_hat = torch.tanh(self.W_C(x))
        ct = ft * pre_cell_state + it + ct_hat
        ot = torch.sigmoid(self.W_o(x))
        ht = ot * torch.tanh(ct)
        return ht, ct


class LSTM(nn.Module):
    '''
    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`

    Inputs: input, (h_0, c_0)
        - **input** of shape `(seq_len, batch, input_size)`: tensor containing the features
          of the input sequence.
          The input can also be a packed variable length sequence.
        - **h_0** of shape `(batch, hidden_size)`: tensor
          containing the initial hidden state for each element in the batch.
        - **c_0** of shape `(batch, hidden_size)`: tensor
          containing the initial cell state for each element in the batch.

          If `(h_0, c_0)` is not provided, both **h_0** and **c_0** default to zero.


    Outputs: output, (h_n, c_n)
        - **output** of shape `(seq_len, batch, hidden_size)`: tensor
          containing the output features `(h_t)` from the last layer of the LSTM,
          for each `t`.
    '''
    def __init__(self, input_size , hidden_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell = LSTMcell(self.input_size, self.hidden_size)

    def forward(self, input, hx=None):
        if hx is None:
            # If(h_0, c_0) is not provided, both h_0 and c_0 default to zero.
            h = torch.zeros(input.size(1), self.hidden_size, dtype=input.dtype, device=input.device)  # [batch,hidden_size]
            c = torch.zeros(input.size(1), self.hidden_size, dtype=input.dtype, device=input.device)
        else:
            h = hx[0]
            c = hx[1]

        output = []
        for x in input:
            h, c = self.cell(x, (h, c))
            output.append(h)
        return torch.stack(output), (h, c)


class PoetryModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(PoetryModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = LSTM(embedding_dim, self.hidden_dim)
        self.linear = nn.Linear(self.hidden_dim, vocab_size)

    def forward(self, input, hidden=None):
        seq_len, batch_size = input.size()
        embeds = self.embeddings(input)
        output, hidden = self.lstm(embeds, hidden)
        output = self.linear(output.view(seq_len * batch_size, -1))
        return output, hidden


if __name__ == "__main__":
    rnn = LSTM(10, 100)
    input = torch.randn(3, 2, 10)
    print(rnn(input).size())