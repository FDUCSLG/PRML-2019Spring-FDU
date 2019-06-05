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

fitlog.commit(__file__)             # auto commit your codes
fitlog.add_hyper_in_file (__file__) # record your hyperparameters

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

#extract k max
class RCNNTextUpdate(torch.nn.Module):
    
    def __init__(self, init_embed,
                 num_classes,
                 hidden_size=64,
                 num_layers=1,
                 linear_hidden_dim=32,
                 kernel_nums=5,
                 kernel_sizes=4,
                 padding=0,
                 content_dim=100,
                 dropout=0.5):
        super(RCNNTextUpdate, self).__init__()
        
        #embedding
        self.embed = encoder.Embedding(init_embed)
        
        #RNN layer
        self.lstm = nn.LSTM(
            input_size=self.embed.embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=True,
            batch_first = False,
            bidirectional=True
        )
        
        
        #CNN layer
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_size*2 + self.embed.embedding_dim,
                out_channels=content_dim,
                kernel_size=kernel_sizes
            ),
            nn.BatchNorm1d(content_dim),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(
                in_channels=content_dim,
                out_channels=content_dim,
                kernel_size=kernel_sizes
            ),
            nn.BatchNorm1d(content_dim),
            nn.ReLU(inplace=True)
        )
        
        #fc
        self.fc = nn.Sequential(
            nn.Linear(content_dim, linear_hidden_dim),
            nn.BatchNorm1d(linear_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(linear_hidden_dim,num_classes)
        )
    
        self.dropout = nn.Dropout(dropout)
#         self.fc = nn.Linear(sum(kernel_nums), num_classes)
    
    def forward(self, words, seq_len=None):
        
        x = self.embed(words) #x: batch, seq, embed
        batch_size_ = x.size()[0]
        lstmOut = self.lstm(x.permute(1,0,2))
        
#         print(lstmOut[0].size())
        lstmOut = lstmOut[0].permute(1,2,0)
        x_new = x.permute(0,2,1) #x_new: batch, embed, seq
        lstmOut = torch.cat((x_new, lstmOut), dim=1)

        convOut = self.conv(lstmOut)

        fcin = self.kmax_pooling(convOut, dim=2)
#         print(fcin.size())
        fcin = fcin.view(fcin.size(0), -1)
        out = self.fc(fcin)
        
        
        return {C.OUTPUT: out}
    
    def kmax_pooling(self,x, dim):
        index = x.topk(1, dim = dim)[1].sort(dim = dim)[0]
        return x.gather(dim, index)
    
    def predict(self, words, seq_len=None):
        output = self(words, seq_len)
        _, predict = output[C.OUTPUT].max(dim=1)
        return {C.OUTPUT: predict}
    
    
    
model = RCNNTextUpdate((len(vocab),128), num_classes=20, padding=1, dropout=0.1)
trainer = Trainer(model=model, train_data=train_data, dev_data=dev_data, loss=CrossEntropyLoss(), metrics=AccuracyMetric(), batch_size=16)
trainer.train()
