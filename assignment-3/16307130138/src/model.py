import numpy as np
import sys
sys.path.append('../')

import torch.nn as nn
from torch.optim import Adam
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from fastNLP import Trainer
from fastNLP.core.losses import CrossEntropyLoss
from fastNLP.core.metrics import AccuracyMetric
from fastNLP import Batch
from copy import deepcopy

from MyUtils.MyLSTM import MyLSTM
from MyUtils.dataset import Config,PoemData
from MyUtils.losses import MyCrossEntropyLoss
from MyUtils.metrics import MyPerplexityMetric

class PoetryModel(nn.Module):
    def __init__(self,vocab_size,conf,device):
        super().__init__()
        self.num_layers = conf.num_layers
        self.hidden_dim = conf.hidden_dim
        self.device = device
        #word embedding layer
        self.embedding = nn.Embedding(vocab_size,conf.embedding_dim)
        # network structure:2 layer lstm
        self.lstm = nn.LSTM(conf.embedding_dim,self.hidden_dim,num_layers = conf.num_layers)
        # 全连接层，后接sofmax进行classification
        self.linear_out = nn.Linear(self.hidden_dim,vocab_size)

    def forward(self, input, hidden=None):
        seq_len,batch_size = input.size()
        # embeds_size = (seq_len*batch_size*embedding_dim)
        embeds = self.embedding(input)
        # print(input.shape())
        if hidden is None:
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device)
            c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device)
        else:
            h_0,c_0 = hidden
        output,hidden = self.lstm(embeds,(h_0,c_0))
        # output_size = (seq_len*batch_size*vocab_size)
        output = self.linear_out(output.view(seq_len*batch_size,-1))
        return output,hidden

class MyPoetryModel(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_dim,device=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.embedding = nn.Embedding(self.vocab_size,self.embedding_dim)
        self.mylstm = MyLSTM(self.embedding_dim,self.hidden_dim,self.hidden_dim)
        self.linear_out = nn.Linear(self.hidden_dim,self.vocab_size)

    def forward(self, input,hidden=None):
        # print("input_size:",input.size())
        seq_len,batch_size = input.size()
        embeds = self.embedding(input)
        output,hidden = self.mylstm(embeds,hidden)
        output = output.view(seq_len*batch_size,-1)
        output = self.linear_out(output)
        # print("Output_size:",output.size())
        return output,hidden 

class FastNLPPoetryModel(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_dim,device=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.device = device

        self.embedding = nn.Embedding(self.vocab_size,self.embedding_dim)
        self.mylstm = MyLSTM(self.embedding_dim,self.hidden_dim,self.hidden_dim)
        self.linear_out = nn.Linear(self.hidden_dim,self.vocab_size)

    def forward(self, input):
        input = input.transpose(0,1)
        seq_len,batch_size = input.size()
        embeds = self.embedding(input)
        output,hidden = self.mylstm(embeds)
        output = output.transpose(0,1).contiguous()
        output = output.view(seq_len*batch_size,-1)
        output = self.linear_out(output)
        return {'output':output,'hidden':hidden}    