import torch
import torch.nn as nn

from fastNLP import DataSet
from fastNLP.modules import encoder
from fastNLP.core import Const

class RNN_model(nn.Module):
    def __init__(self, dict_size, embedding_dim, hidden_dim, num_classes):
        super(RNN_model, self).__init__()
        self.embedding = nn.Embedding(dict_size, embedding_dim)
        self.rnn = encoder.LSTM(
            input_size = self.embedding.embedding_dim,
            hidden_size = hidden_dim,
            num_layers = 1,
        )
        self.output = nn.Linear(hidden_dim, num_classes)

    def forward(self, words):
        x = self.embedding(words)
        r_output, _ = self.rnn(x, None)
        mean = r_output.sum(1) / r_output.shape[1]
        output = self.output(mean)
        return {Const.OUTPUT: output}

    def predict(self, words):
        output = self(words)
        _, predict = output['pred'].max(dim=1)
        return {'pred': predict}
