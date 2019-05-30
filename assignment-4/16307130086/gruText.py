import os
import fastNLP
os.sys.path.append('..')
# from handout import get_text_classification_datasets
# trainData,testData = get_text_classification_datasets()
from fastNLP import Instance
from fastNLP import DataSet
from fastNLP import Vocabulary
# from fastNLP.models import CNNText
from fastNLP import Trainer, CrossEntropyLoss, AccuracyMetric
from sklearn.datasets import fetch_20newsgroups
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from fastNLP.core.const import Const as C
from fastNLP.modules import encoder

trainSet = fetch_20newsgroups(subset='train')
testSet = fetch_20newsgroups(subset='test')

testData = {
    "data": testSet['data'],
    "target": testSet['target']
}
trainData = {
    "data": trainSet['data'],
    "target": trainSet['target']
}
trainData = DataSet(trainData)
testData = DataSet(testData)

trainData.apply(lambda x: x['data'].lower(), new_field_name='sentence')
trainData.apply(lambda x: x['sentence'].split(), new_field_name='words', is_input=True)
vocab = Vocabulary(min_freq=2)
vocab = vocab.from_dataset(trainData, field_name='words')
#change to index
vocab.index_dataset(trainData, field_name='words',new_field_name='words')
trainData.set_target('target')
train_data, dev_data = trainData.split(0.2)

#demo LSTM version
class GRUtext(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, hidden_dim=64, num_layers=2, dropout=0.5):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # note: bidirectional leading to hidden num_layers*2
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, words):
        
        # (input) words : (batch_size, seq_len)
        words = words.permute(1,0)
        # words : (seq_len, batch_size)

        embedded = self.dropout(self.embedding(words))
        # embedded : (seq_len, batch_size, embedding_dim)
        output, hidden = self.gru(embedded)
        
        # output: (seq_len, batch_size, hidden_dim * 2)
        # hidden: (num_layers * 2, batch_size, hidden_dim)
        # cell: (num_layers * 2, batch_size, hidden_dim)

        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        hidden = self.dropout(hidden)
        # hidden: (batch_size, hidden_dim * 2)

        pred = self.fc(hidden.squeeze(0))
        # result: (batch_size, output_dim)
        return {"pred":pred}

loss = CrossEntropyLoss()
metrics=AccuracyMetric()
model_lstm = GRUtext(len(vocab), embedding_dim=128, output_dim=20, dropout=0.1)
trainer = Trainer(model=model_lstm, train_data=train_data, dev_data=dev_data,loss=loss, metrics=metrics)
trainer.train()