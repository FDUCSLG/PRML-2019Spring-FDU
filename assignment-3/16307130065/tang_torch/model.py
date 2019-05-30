import torch
import torch.nn as nn
from torch.nn import Parameter
import numpy as np


class PoetryModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # input gate
        self.W_xi = Parameter(torch.rand(embedding_dim, hidden_dim)*np.sqrt(2/(embedding_dim + hidden_dim)))
        self.W_hi = Parameter(torch.rand(hidden_dim, hidden_dim)*np.sqrt(2/(2*hidden_dim)))
        self.b_i = Parameter(torch.rand(1, hidden_dim))
        # forget gate
        self.W_xf = Parameter(torch.rand(embedding_dim, hidden_dim)*np.sqrt(2/(embedding_dim + hidden_dim)))
        self.W_hf = Parameter(torch.rand(hidden_dim, hidden_dim)*np.sqrt(2/(2*hidden_dim)))
        self.b_f = Parameter(torch.rand(1, hidden_dim))
        # cell update
        self.W_xc = Parameter(torch.rand(embedding_dim, hidden_dim)*np.sqrt(2/(embedding_dim + hidden_dim)))
        self.W_hc = Parameter(torch.rand(hidden_dim, hidden_dim)*np.sqrt(2/(2*hidden_dim)))
        self.b_c = Parameter(torch.rand(1, hidden_dim))
        # output gate
        self.W_xo = Parameter(torch.rand(embedding_dim, hidden_dim)*np.sqrt(2/(embedding_dim + hidden_dim)))
        self.W_ho = Parameter(torch.rand(hidden_dim, hidden_dim)*np.sqrt(2/(2*hidden_dim)))
        self.b_o = Parameter(torch.rand(1, hidden_dim))

        self.W = nn.Parameter(torch.rand(hidden_dim, vocab_size)*np.sqrt(2/(hidden_dim+vocab_size)))
        self.b = nn.Parameter(torch.rand(1, vocab_size))

    def forward(self, input_, hidden=None):
        seq_len, batch_size = input_.size()
        if hidden is None:
            h_t = input_.data.new(batch_size, self.hidden_dim).fill_(0).float()
            c_t = input_.data.new(batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_t, c_t = hidden

        embeds = self.embeddings(input_)

        tmp = []
        for t in range(seq_len):
            x_t = embeds[t]
            i_t = torch.sigmoid(x_t.mm(self.W_xi) + h_t.mm(self.W_hi) + self.b_i)
            f_t = torch.sigmoid(x_t.mm(self.W_xf) + h_t.mm(self.W_hf) + self.b_f)
            c_t_ = torch.tanh(x_t.mm(self.W_xc) + h_t.mm(self.W_hc) + self.b_c)
            o_t = torch.sigmoid(x_t.mm(self.W_xo) + h_t.mm(self.W_ho) + self.b_o)
            c_t = f_t * c_t + i_t * c_t_
            h_t = o_t * torch.tanh(c_t)
            opt = h_t.mm(self.W) + self.b
            tmp.append(opt)
        output = torch.cat(tmp, 0)
        return output, (h_t, c_t)
