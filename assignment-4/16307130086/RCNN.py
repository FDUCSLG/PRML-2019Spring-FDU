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

# 
#Siwei Lai. 2015. Recurrent Convolutional Neural Networks for Text Classification
class RCNNText(torch.nn.Module):
    
    def __init__(self, init_embed,
                 num_classes,
                 hidden_size=64,
                 num_layers=1,
                 kernel_nums=(3, 4, 5),
                 kernel_sizes=(3, 4, 5),
                 padding=0,
                 dropout=0.5):
        super(RCNNText, self).__init__()
        
        #embedding
        self.embed = encoder.Embedding(init_embed)
        
        self.conv_pool = encoder.ConvMaxpool(
            in_channels=self.embed.embedding_dim+hidden_size*2,
            out_channels=kernel_nums,
            kernel_sizes=kernel_sizes,
            padding=padding)
#         #RNN layer
        self.lstm = nn.LSTM(
            input_size=self.embed.embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=True,
            bidirectional=True
        )
    
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(sum(kernel_nums), num_classes)
    
    def forward(self, words, seq_len=None):
        x = self.embed(words) 
        output, (hidden, cell) = self.lstm(x)
#         print("output",output.size())
        batch_size_ = x.size()[0]
        
        x = torch.cat((output, x), dim=2)
#         print("y",x.size())
        x = self.conv_pool(x)
        x = self.dropout(x)
        x = self.fc(x)
        return {C.OUTPUT: x}
    
    def predict(self, words, seq_len=None):
        output = self(words, seq_len)
        _, predict = output[C.OUTPUT].max(dim=1)
        return {C.OUTPUT: predict}
    
    
model = RCNNText((len(vocab),128), num_classes=20, padding=2, dropout=0.1)
trainer = Trainer(model=model, train_data=train_data, n_epochs=20, dev_data=dev_data, loss=CrossEntropyLoss(), metrics=AccuracyMetric(), batch_size=16)
trainer.train()

