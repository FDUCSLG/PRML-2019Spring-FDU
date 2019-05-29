import torch
import torch.nn as nn

from fastNLP.core.const import Const as C
from fastNLP.modules import encoder


class CNN_model(torch.nn.Module):
    
    def __init__(self, dict_size, embedding_dim, num_classes,
                 kernel_nums=(3, 4, 5), kernel_sizes=(3, 4, 5),
                 padding=0, dropout=0.5):
        super(CNN_model, self).__init__()

        self.embedding_dim = embedding_dim
        self.embed = nn.Embedding(dict_size, embedding_dim)
        
        self.conv_pool = encoder.ConvMaxpool(
            in_channels = self.embedding_dim,
            out_channels = kernel_nums, kernel_sizes = kernel_sizes,
            padding = padding)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(sum(kernel_nums), num_classes)
        
    def forward(self, words, seq_len=None):


        x = self.embed(words)
        x = self.conv_pool(x)
        x = self.dropout(x)
        x = self.fc(x)
        return {C.OUTPUT: x}
    
    def predict(self, words, seq_len=None):

        output = self(words, seq_len)
        _, predict = output[C.OUTPUT].max(dim = 1)
        return {C.OUTPUT: predict}
