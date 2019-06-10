# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from fastNLP import Const


class BiRNNText_glove(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, hidden_dim=64, num_layers=2, dropout=0.5):
        super(BiRNNText_glove, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.constant_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(2 * embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=True, 
                            batch_first=True, dropout=dropout)
                            # batch_first: input and output tensors are provided as (batch, seq, feature).
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers

    def forward(self, words):
        # (input) words : (batch_size, seq_len)
        # print(words.shape)
        embedded = torch.cat((self.embedding(words), self.constant_embedding(words)), dim=2)
        embedded = self.dropout(embedded)
        # embedded : (batch_size, seq_len, 2 * embedding_dim)
        output, (hidden, cell) = self.lstm(embedded)
        # output: (batch_size, seq_len, hidden_dim * 2)
        # hidden: (num_layers * 2, batch_size, hidden_dim)
        # cell: (num_layers * 2, batch_size, hidden_dim)

        # dropout
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        hidden = self.dropout(hidden)
        # hidden: (batch_size, hidden_dim * 2)

        pred = self.fc(hidden)
        # pred: (batch_size, output_dim)
        return {Const.OUTPUT:pred}

    def predict(self, words):
        output = self(words)
        _, predict = output[Const.OUTPUT].max(dim=1)
        return {Const.OUTPUT:predict}


# Convolutional Neural Networks for Sentence Classification 
class TextCNN_glove(nn.Module):
    def __init__(self, vocab_size, embedding_dim, kernel_sizes, num_channels, num_classes, activation='relu',
                    dropout=0.5, bias=True, stride=1, padding=0, dilation=1, groups=1):
        super(TextCNN_glove, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.constant_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        # 创建多个一维卷积层
        self.convs = nn.ModuleList([nn.Conv1d(
                in_channels=2 * embedding_dim,
                out_channels=oc,
                kernel_size=ks,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias)
                for oc, ks in zip(num_channels, kernel_sizes)])
        
        # activation function
        if activation == 'relu':
            self.activation = nn.functional.relu
        elif activation == 'sigmoid':
            self.activation = nn.functional.sigmoid
        elif activation == 'tanh':
            self.activation = nn.functional.tanh
        else:
            raise Exception(
                "Undefined activation function: choose from: relu, tanh, sigmoid")
        
        self.fc = nn.Linear(sum(num_channels), num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, words):
        # (input) words : (batch_size, seq_len)
        # print(words.shape)
        embedded = torch.cat((self.embedding(words), self.constant_embedding(words)), dim=2)
        # embedded = self.dropout(embedded)
        embedded = embedded.permute(0, 2, 1)
        # embedded : (batch_size, 2 * embedding_dim, seq_len)
        
        # convolution
        xs = [self.activation(conv(embedded)) for conv in self.convs]
        # max-pooling
        xs = [nn.functional.max_pool1d(input=x, kernel_size=x.size(2)).squeeze(2)
              for x in xs]
        # x : (batch_size, num_channels)

        encoding = torch.cat(xs, dim=-1)

        # Dropout
        encoding = self.dropout(encoding)
        pred = self.fc(encoding)
        return {Const.OUTPUT:pred}

    def predict(self, words):
        output = self(words)
        _, predict = self.softmax(output[Const.OUTPUT]).max(dim=1)
        return {Const.OUTPUT:predict}