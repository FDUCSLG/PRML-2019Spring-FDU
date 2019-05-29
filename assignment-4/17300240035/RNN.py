import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import fastNLP.modules.encoder as encoder


class RNN_Text(torch.nn.Module):

    def __init__(self, vocab_size, input_size, hidden_layer_size, target_size, dropout):
        super(RNN_Text, self).__init__()

        self.embed = encoder.Embedding((vocab_size, input_size))
        self.LSTM = encoder.lstm.LSTM(input_size=input_size, hidden_size=hidden_layer_size)
        self.dropout = nn.Dropout(dropout)
        self.hidden2tag = nn.Linear(hidden_layer_size, target_size)

    def forward(self, text):
        input = self.embed(text)
        x, _ = self.LSTM(input)
        x = self.dropout(_[0][-1])
        x = self.hidden2tag(x)
        return {'pred': x}

    def predict(self, text):
        output = self(text)
        predict = output['pred'].argmax(dim=1)
        return {'pred': predict}
