# coding: utf-8
from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# x = [sentence length, batch size]

class RNN(nn.Module):
    def __init__(self, input_dim, params):
        super(RNN, self).__init__()
        embedding_dim = params['embedding_dim']
        hidden_dim = params['hidden_dim']
        output_dim = params['output_dim']

        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)
        assert torch.equal(output[-1, :, :], hidden.squeeze(0))

        return self.fc(hidden.squeeze(0))

class LSTM(nn.Module):
    def __init__(self, vocab_size, params):
        super(LSTM, self).__init__()
        embedding_dim = params['embedding_dim']
        hidden_dim = params['hidden_dim']
        output_dim = params['output_dim']
        n_layers = params['n_layers']
        bidirectional = params['bidirectional']
        dropout = params['dropout']

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout)
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        embedded = self.dropout(self.embedding(x))    
        output, (hidden, cell) = self.lstm(embedded) 
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        return self.fc(hidden.squeeze(0))

class FastText(nn.Module):
    def __init__(self, vocab_size, params):
        super(FastText, self).__init__()
        embedding_dim = params['embedding_dim']
        output_dim = params['output_dim']
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, output_dim)
        
    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.permute(1, 0, 2)
        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1)       
        return self.fc(pooled)
        #output = self.fc(pooled)
        #return {'output': output}

class CNN(nn.Module):
    def __init__(self, vocab_size, params):
        super(CNN, self).__init__()
        embedding_dim = params['embedding_dim']
        n_filters = params['n_filters']
        filter_sizes = params['filter_sizes']
        output_dim = params['output_dim']
        dropout = params['dropout']
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs,embedding_dim)) for fs in filter_sizes])
        self.fc = nn.Linear(len(filter_sizes)*n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = x.permute(1, 0)
        embedded = self.embedding(x)
        embedded = embedded.unsqueeze(1)
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)
