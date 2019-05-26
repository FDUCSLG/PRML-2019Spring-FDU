import torch
import torch.nn as nn
from torch.nn import Parameter
import numpy as np


class PoetryModel(nn.Module):
    def __init__(self, dict_size, embedding_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(dict_size, embedding_dim)
        # input gate
        self.W_xi, self.W_hi, self.b_i = self.init_gate(embedding_dim, hidden_dim)
        # forget gate
        self.W_xf, self.W_hf, self.b_f = self.init_gate(embedding_dim, hidden_dim)
        # cell update
        self.W_xc, self.W_hc, self.b_c = self.init_gate(embedding_dim, hidden_dim)
        # output gate
        self.W_xo, self.W_ho, self.b_o  = self.init_gate(embedding_dim, hidden_dim)

        self.W = Parameter(torch.rand(hidden_dim, dict_size) * np.sqrt(2/(hidden_dim + dict_size)))
        self.b = Parameter(torch.rand(1, dict_size))

    def init_gate(self, embedding_dim, hidden_dim):
        x = Parameter(torch.rand(embedding_dim, hidden_dim) * np.sqrt(2 / (embedding_dim + hidden_dim)))
        h = Parameter(torch.rand(hidden_dim, hidden_dim) * np.sqrt(2 / (2 * hidden_dim)))
        b = Parameter(torch.rand(1, hidden_dim))

        return x, h, b
        
    def forward(self, X, hidden=None):
        seq_len, batch_size = X.size()
        if hidden is None:
            ht = X.data.new(batch_size, self.hidden_dim).fill_(0).float()
            ct = X.data.new(batch_size, self.hidden_dim).fill_(0).float()
        else:
            ht, ct = hidden

        embeds = self.embed(X)

        tmp = []
        for t in range(seq_len):
            xt = embeds[t]
            it = torch.sigmoid(xt.mm(self.W_xi) + ht.mm(self.W_hi) + self.b_i)
            ft = torch.sigmoid(xt.mm(self.W_xf) + ht.mm(self.W_hf) + self.b_f)
            cit = torch.tanh(xt.mm(self.W_xc) + ht.mm(self.W_hc) + self.b_c)
            ot = torch.sigmoid(xt.mm(self.W_xo) + ht.mm(self.W_ho) + self.b_o)
            ct = ft * ct + it * cit
            ht = ot * torch.tanh(ct)
            opt = ht.mm(self.W) + self.b
            tmp.append(opt)
        output = torch.cat(tmp, 0)
        return output, (ht, ct)
