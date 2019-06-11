from sklearn.datasets import fetch_20newsgroups
import torch
import numpy as np
from fastNLP import CrossEntropyLoss,AccuracyMetric
import torch.nn as nn
from torch.nn import functional as F
from fastNLP.io import CSVLoader
from fastNLP import Vocabulary,Instance,DataSet
from fastNLP import Const,Trainer,Tester
from fastNLP.models import CNNText
from fastNLP.modules import encoder
import string
from fastNLP import BucketSampler
from fastNLP import Batch
from fastNLP.core.callback import FitlogCallback
import fitlog

class mycnn(nn.Module):
    def __init__(self, vocab_len, embed_dim, 
                 num_classes, 
                 kernel_num = 16,
                 kernel_sizes=(3,4,5),
                 padding=0,
                 dropout=0.5):
        super(mycnn, self).__init__()
        ci = 1
        self.embed = nn.Embedding(vocab_len, embed_dim)
        self.convl1 = nn.Conv2d(ci,kernel_num,(kernel_sizes[0],embed_dim))
        self.convl2 = nn.Conv2d(ci,kernel_num,(kernel_sizes[1],embed_dim))
        self.convl3 = nn.Conv2d(ci,kernel_num,(kernel_sizes[2],embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_sizes) * kernel_num, num_classes)

    def conv_and_pool(self,x,conv):
        x = conv(x)
        x = F.relu(x.squeeze(3))
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x
    
    def forward(self, words):
        x = self.embed(words)
        x = x.unsqueeze(1)
        x1 = self.conv_and_pool(x, self.convl1)
        x2 = self.conv_and_pool(x, self.convl2)
        x3 = self.conv_and_pool(x, self.convl3)
        x = torch.cat((x1,x2,x3), 1)
        x = self.dropout(x)
        #logit = F.log_softmax(self.fc(x), dim=1)
        logit = self.fc(x)
        return {"pred":logit}

class myrnn(nn.Module):
    def __init__(self, vocab_size, embed_dim, class_num, hidden_dim=64, dropout=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(hidden_dim*2,class_num) 
        #self.W = nn.Parameter(torch.rand(hidden_dim,embed_dim),requires_grad=True)
        #self.U = nn.Parameter(torch.rand(hidden_dim,hidden_dim),requires_grad=True)
        #self.b = nn.Parameter(torch.rand(hidden_dim,1),requires_grad=True)
        self.rnn = nn.RNN(input_size = embed_dim,hidden_size = hidden_dim, num_layers = 2, bidirectional=True , nonlinearity = 'relu')
       
        self.dropout = nn.Dropout(dropout)

    def forward(self, words):
        words = words.permute(1,0)
        embed = self.dropout(self.embedding(words))
        
        output,hidden = self.rnn(embed)
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        hidden = self.dropout(hidden)
        # hidden: (batch_size, hidden_dim * 2)

        pred = self.fc(hidden.squeeze(0))
        # result: (batch_size, output_dim)
        return {"pred":pred}

class LSTMText(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, hidden_dim=64, num_layers=2, dropout=0.5):
        super().__init__()
        #self.W = nn.Parameter(torch.rand(hidden_dim,embed_dim),requires_grad=True)
        #self.U = nn.Parameter(torch.rand(hidden_dim,hidden_dim),requires_grad=True)
        #self.b = nn.Parameter(torch.rand(hidden_dim,1),requires_grad=True)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, words):
        
        words = words.permute(1,0)

        embedded = self.dropout(self.embedding(words))

        output, (hidden, cell) = self.lstm(embedded)

        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)

        hidden = self.dropout(hidden)

        pred = self.fc(hidden.squeeze(0))

        return {"pred":pred}

# Prepare the dataset and testset
fitlog.commit(__file__)
fitlog.add_hyper_in_file(__file__)

table = str.maketrans('','',string.punctuation)
newsgroups_train = fetch_20newsgroups(subset='train')
dataset = DataSet()
for i in range(newsgroups_train.target.shape[0]):
    dataset.append(Instance(raw_sentence=newsgroups_train.data[i].replace('\n',' '),target=int(newsgroups_train.target[i])))
dataset.apply(lambda x: x['raw_sentence'].lower().translate(table), new_field_name='sentence')
dataset.apply_field(lambda x: x.split(), field_name='sentence', new_field_name='words')
dataset.apply_field(lambda x: len(x), field_name='words', new_field_name='seq_len')

newsgroups_test = fetch_20newsgroups(subset='test')
testset = DataSet()
for i in range(newsgroups_test.target.shape[0]):
    testset.append(Instance(raw_sentence=newsgroups_test.data[i].replace('\n',' '),target=int(newsgroups_test.target[i])))
testset.apply(lambda x: x['raw_sentence'].lower().translate(table), new_field_name='sentence')
testset.apply_field(lambda x: x.split(), field_name='sentence', new_field_name='words')
testset.apply_field(lambda x: len(x), field_name='words', new_field_name='seq_len')

vocab = Vocabulary(min_freq=10).from_dataset(dataset, field_name='words')
vocab.index_dataset(dataset, field_name='words', new_field_name='words')
vocab.index_dataset(testset, field_name='words', new_field_name='words')

#model = CNNText((len(vocab),50), num_classes=20, padding=2, dropout=0.1)
model = mycnn(len(vocab),100,len(dataset.target))
#model = myrnn(len(vocab),100,20)
#model = LSTMText(len(vocab),64,20) #used

dataset.rename_field('words', Const.INPUT)
dataset.rename_field('target', Const.TARGET)
dataset.rename_field('seq_len', Const.INPUT_LEN)
dataset.set_input(Const.INPUT, Const.INPUT_LEN)
dataset.set_target(Const.TARGET)

testset.rename_field('words', Const.INPUT)
testset.rename_field('target', Const.TARGET)
testset.rename_field('seq_len', Const.INPUT_LEN)
testset.set_input(Const.INPUT, Const.INPUT_LEN)
testset.set_target(Const.TARGET)

train_data, dev_data = dataset.split(0.1)

loss = CrossEntropyLoss(pred=Const.OUTPUT, target=Const.TARGET)
metrics = AccuracyMetric(pred=Const.OUTPUT, target=Const.TARGET)
trainer = Trainer(model=model, train_data=train_data, dev_data=dev_data, loss=loss, batch_size=16, metrics=metrics, n_epochs=20 ,callbacks=[FitlogCallback(dataset)])
trainer.train()

tester = Tester(data = testset, model = model, metrics = metrics)
tester.test()

tester = Tester(data = train_data, model = model, metrics = metrics)
tester.test()

fitlog.finish()
