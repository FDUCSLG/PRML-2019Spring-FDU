import numpy as np
import torch
import torch.nn as nn
from fastNLP.modules import encoder
from fastNLP.core.const import Const

class CNN_model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, kernel_nums=(4,5,6), kernel_sizes=(4,5,6),
                 padding=2, dropout=0.1):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv_pool = encoder.ConvMaxpool(
            in_channels=self.embedding.embedding_dim,
            out_channels=kernel_nums,
            kernel_sizes=kernel_sizes,
            padding=padding)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(sum(kernel_nums), num_classes)

    def forward(self, words, seq_len = None):
        x = self.embedding(words)
        x = self.conv_pool(x)
        x = self.dropout(x)
        x = self.fc(x)
        return {Const.OUTPUT: x}
