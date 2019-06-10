import os
import sys
import pickle
import argparse
import logging
import fitlog

sys.path.append('../')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from fastNLP.models import CNNText,STSeqCls
from fastNLP.core.const import Const as C
from fastNLP.modules import Embedding as fastEmbeds
class MyCNNText(nn.Module):
    def __init__(self, init_embeds,num_classes,
                 kernel_nums=(3, 4, 5),kernel_sizes=(3, 4, 5),
                 padding=0,dropout=0.5):
        super(MyCNNText,self).__init__()
        self.embeds = fastEmbeds(init_embeds,padding_idx=0)
        vocab_size,embedding_dim = self.embeds.weight.size()

        self.embeds = nn.Embedding(vocab_size,embedding_dim)
        in_c = embedding_dim
        if isinstance(kernel_nums,int) and isinstance(kernel_sizes,int) :
            kernel_nums = [kernel_nums]
            kernel_sizes = [kernel_sizes]
        self.convs1 = nn.ModuleList([nn.Conv2d(1,out_c,(ks,embedding_dim) )
                        for out_c, ks in zip(kernel_nums, kernel_sizes)])
        self.convs2 = nn.ModuleList([nn.Conv1d(
                        in_channels=in_c,
                        out_channels=out_c,
                        kernel_size=ks,
                        padding=padding)
                        for out_c, ks in zip(kernel_nums, kernel_sizes)])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(sum(kernel_nums),num_classes)
    
    def conv_and_pool(self,x,conv):
        x = F.relu(conv(x).squeeze(3))
        x = F.max_pool1d(x,x.size(2)).squeeze(2)
        return x

    def forward(self,words,seq_len=None):
        x = self.embeds(words)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
        x = [F.max_pool1d(i,i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x,1)
        x = self.dropout(x)
        x = self.fc(x)
        return {C.OUTPUT:x}

    def predict(self,words,seq_len=None):
        output = self(words, seq_len)
        _, predict = output[C.OUTPUT].max(dim=1)
        return {C.OUTPUT: predict}

class MySTSeqCls(nn.Module):
    def __init__(self):
        super().__init__()

class RNNText(nn.Module):
    def __init__(self):
        super().__init__()

class LSTMText(nn.Module):
    def __init__(self, init_embeds, output_dim, hidden_dim=64, num_layers=2, dropout=0.5):
        super().__init__()
        # if isinstance(init_embeds,tuple):
        #     vocab_size, embedding_dim = init_embeds
        #     self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embeds = fastEmbeds(init_embeds)
        num_embeddings, embedding_dim = self.embeds.weight.size()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, words):
        # (input) words : (batch_size, seq_len)
        words = words.permute(1,0)
        # words : (seq_len, batch_size)

        embedded = self.dropout(self.embeds(words))
        # embedded : (seq_len, batch_size, embedding_dim)
        output, (hidden, cell) = self.lstm(embedded)
        # output: (seq_len, batch_size, hidden_dim * 2)
        # hidden: (num_layers * 2, batch_size, hidden_dim)
        # cell: (num_layers * 2, batch_size, hidden_dim)

        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        hidden = self.dropout(hidden)
        # hidden: (batch_size, hidden_dim * 2)

        pred = self.fc(hidden.squeeze(0))
        # result: (batch_size, output_dim)
        return {"pred":pred}