import torch
import torch.nn as nn
from torch.nn import Parameter
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class RNNcell(nn.Module):
  def __init__(input_dim, hidden_dim):
    super().__init__()
    self.input_dim, self.hidden_dim = input_dim, hidden_dim
    self.W_ih = Parameter(torch.Tensor(input_dim, hidden_dim))
    self.W_hh = Parameter(torch.Tensor(hidden_dim, hidden_dim))
    self.bias_hh = Parameter(torch.Tensor(hidden_dim))
    self.init_weights()

  def init_weights(self):
    nn.init.xavier_uniform_(self.W_ih)
    nn.init.xavier_uniform_(self.W_hh)
    nn.init.zeros_(self.bias_hh)

  def forward(self, input, init_state):
    # input: [seq_len, batch_size, hidden_dim]
    seq_len = len(input)
    hidden_seq = []
    h_t = init_state
    for t in range(seq_len):
      x_t = input[t]
      h_t = torch.tanh(x_t @ self.W_ih + h_t @ self.W_hh + self.bias_hh)
      hidden_seq.append(h_t)
    hidden_seq = torch.cat(hidden_seq).view(seq_len, -1, self.hidden_dim)
    # only return the last: [batch_size, hidden_dim]
    return hidden_seq[-1], h_t

class RNN(nn.Module):
  def __init__(vocab_size, input_dim, hidden_dim, output_dim):
    super().__init__()
    self.vocab_size = vocab_size
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.output_dim = output_dim

    self.embed = nn.Embedding(vocab_size, input_dim)
    self.rnn = RNNcell(input_dim, hidden_dim)
    self.linear = nn.linear(hidden_dim, output_dim)

  def init_hidden(self, batch_size=32):
    self.hidden = torch.zeros(batch_size, self.hidden_dim)

  def forward(self, input):
    # input: [batch_size, seq_len] -> [batch_size, seq_len, input_dim] -> [seq_len, batch_size, input_dim]
    input = self.embed(input).permute(1, 0, 2)
    # forward
    rnn_out, self.hidden = self.rnn(input, self.hidden)
    y_pred = self.linear(rnn_out)
    # output: [batch_size, output_dim]
    return { 'output': y_pred }
