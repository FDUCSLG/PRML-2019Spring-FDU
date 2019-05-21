import torch
import torch.nn as nn
from torch.nn import Parameter
import numpy as np

class RNNcell(nn.Module):
  def __init__(self, input_dim, hidden_dim):
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
  def __init__(self, vocab_size, input_dim, hidden_dim, output_dim):
    super().__init__()
    self.vocab_size = vocab_size
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.output_dim = output_dim

    self.embed = nn.Embedding(vocab_size, input_dim)
    self.rnn = RNNcell(input_dim, hidden_dim)
    self.linear = nn.Linear(hidden_dim, output_dim)

  def init_hidden(self, batch_size=32):
    self.hidden = torch.zeros(batch_size, self.hidden_dim)

  def forward(self, word_seq):
    # init
    self.init_hidden(batch_size=len(word_seq))
    # word_seq: [batch_size, seq_len] -> [batch_size, seq_len, word_seq_dim] -> [seq_len, batch_size, word_seq_dim]
    word_seq = self.embed(word_seq).permute(1, 0, 2)
    # forward
    rnn_out, self.hidden = self.rnn(word_seq, self.hidden)
    y_pred = self.linear(rnn_out)
    # output: [batch_size, output_dim]
    return { 'output': y_pred, 'pred': nn.functional.softmax(y_pred, dim=1), 'hidden': self.hidden }


class CNN(nn.Module):
  def __init__(self, vocab_size, input_dim, output_dim, in_channels, out_channels,
                kernel_sizes, keep_probab, embedding_weights=None):
    super().__init__()
    self.vocab_size = vocab_size
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_sizes = kernel_sizes

    self.embed = nn.Embedding(vocab_size, input_dim)
    # self.embed.weight = nn.Parameter(embedding_weights, requires_grad=False)
    self.conv1 = nn.Conv2d(in_channels, out_channels, (kernel_sizes[0], input_dim))
    self.conv2 = nn.Conv2d(in_channels, out_channels, (kernel_sizes[1], input_dim))
    self.conv3 = nn.Conv2d(in_channels, out_channels, (kernel_sizes[2], input_dim))
    self.dropout = nn.Dropout(keep_probab)
    self.linear = nn.Linear(len(kernel_sizes)*out_channels, output_dim)

  def conv_calc(self, input, conv_fn):
    conv_out = conv_fn(input)  # conv_out: [batch_size, out_channels, dim, 1]
    conv_relu = nn.functional.relu(conv_out.squeeze(3))  # conv_relu: [batch_size, out_channels, dim1]
    max_out = nn.functional.max_pool1d(conv_relu, conv_relu.size(2)).squeeze(2)  # max_out: [batch_size, out_channels]
    return max_out

  def forward(self, word_seq, batch_size=None):
    # input: [batch_size, num_seq] -> [batch_size, 1, num_seq, embedding_length]
    input = self.embed(word_seq)
    input = input.unsqueeze(1)
    # max_out: [batch_size, out_channels]
    max_out1 = self.conv_calc(input, self.conv1)
    max_out2 = self.conv_calc(input, self.conv2)
    max_out3 = self.conv_calc(input, self.conv3)
    # all_out: [batch_size, num_kernels*out_channels]
    all_maxout = torch.cat((max_out1, max_out2, max_out3), 1)
    all_dropout = self.dropout(all_maxout)
    # logits: [batch_size, output_dim]
    logits = self.linear(all_dropout)
    return { 'output': logits, 'pred': nn.functional.softmax(logits, dim=1) }