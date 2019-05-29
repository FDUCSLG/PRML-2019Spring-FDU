import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import fastNLP.modules.encoder as encoder
from fastNLP.core.const import Const as C


class CNN_Text(torch.nn.Module):

    def __init__(self, vocab_size, input_size, target_size, kernel_num=(3, 4, 5), kernel_size=(3, 4, 5), padding=0, dropout=0.2):
        super(CNN_Text, self).__init__()

        self.embed = encoder.Embedding((vocab_size, input_size))
        self.conv = encoder.ConvMaxpool(in_channels=input_size, out_channels=kernel_num, kernel_sizes=kernel_size, padding=padding)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(sum(kernel_num), target_size)

    def forward(self, text):
        # print(text)
        x = self.embed(text)
        x = self.conv(x)
        x = self.dropout(x)
        x = self.linear(x)
        return {'pred': x}

    def predict(self, text):
        output = self(text)
        predict = output['pred'].argmax(dim=1)
        return {'pred': predict}