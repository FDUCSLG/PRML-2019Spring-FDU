import numpy as np
import torch
import torch.nn as nn
from fastNLP.modules import encoder
from fastNLP.core.const import Const
from pprint import pprint

class RNN_single_gate(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, hidden_dim, dropout=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.wx = nn.Parameter(torch.rand(embedding_dim, hidden_dim) * np.sqrt(2/(embedding_dim+hidden_dim)))
        self.wh = nn.Parameter(torch.rand(hidden_dim, hidden_dim) * np.sqrt(2 / (2 * hidden_dim)))
        self.b = nn.Parameter(torch.rand(1, hidden_dim))

        self.wy = nn.Parameter(torch.rand(hidden_dim, output_dim) * np.sqrt(2/(hidden_dim+output_dim)))
        self.wb = nn.Parameter(torch.rand(1, output_dim))

    def forward(self, words, hidden = None):
        words = words.t()
        sequence, batch = words.size()
        if hidden is None:
            ht = words.data.new(batch, self.hidden_dim).fill_(0).float()
        else:
            ht = hidden
        embeds = self.embedding(words)

        for t in range(sequence):
            xt = embeds[t]
            ht = torch.tanh(xt.mm(self.wx) + ht.mm(self.wh) + self.b)
            Y = ht.mm(self.wy) + self.wb
        return {Const.OUTPUT: Y}



class RNN_model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(RNN_model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = encoder.LSTM(
            input_size=self.embedding.embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
        )
        self.output = nn.Linear(hidden_dim, num_classes)

    def forward(self, words):
        x = self.embedding(words)
        r_output, _ = self.rnn(x, None)
        mean = r_output.sum(1)/r_output.shape[1]
        output = self.output(mean)
        return {Const.OUTPUT: output}
