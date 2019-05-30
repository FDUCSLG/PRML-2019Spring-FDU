import os
os.sys.path.append('..')
#from handout import get_linear_seperatable_2d_2c_dataset
#from handout import get_text_classification_datasets
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from sklearn.datasets import fetch_20newsgroups
import string
import re
import collections
from matplotlib import pyplot as plt
from fastNLP import DataSet
from fastNLP import Instance
from fastNLP import Vocabulary
import torch
import torch.nn as nn
import fastNLP.modules.encoder as encoder
from fastNLP import Trainer
from copy import deepcopy
from fastNLP.core.losses import CrossEntropyLoss
from fastNLP.core.metrics import AccuracyMetric

from fastNLP import Const
from fastNLP import AccuracyMetric
from fastNLP import CrossEntropyLoss
from fastNLP import BucketSampler
from fastNLP import Batch
import torch
import time
import fitlog
from fastNLP.core.callback import FitlogCallback
from fastNLP import Tester
from fastNLP import Callback

fitlog.commit('__file__')             # auto commit your codes
fitlog.add_hyper_in_file ('__file__') # record your hyperparameters
loss = CrossEntropyLoss(pred=Const.OUTPUT, target=Const.TARGET)
metrics=AccuracyMetric(pred=Const.OUTPUT, target=Const.TARGET)
target_len = 20
def readdata():
    global target_len
    min_count = 10
    #categories = ['comp.os.ms-windows.misc', 'rec.motorcycles', 'sci.space', 'talk.politics.misc', ]
    dataset_train = fetch_20newsgroups(subset='train', data_home='../../..')
    dataset_test = fetch_20newsgroups(subset='test', data_home='../../..')

    data = dataset_train.data
    target = dataset_train.target
    target_len = len(dataset_train.target_names)
    train_data =  DataSet()
    padding = 0
    for i in range(len(data)):
        data_t =  re.sub("\d+|\s+|/", " ", data[i] )
        temp = [word.strip(string.punctuation).lower() for word in data_t.split() if word.strip(string.punctuation) != '']
        train_data.append(Instance(raw = data[i], label = int(target[i]), words = temp))
        if len(temp) > padding:
            padding = len(temp)
    train_data.apply(lambda x: x['raw'].lower(), new_field_name='raw')

    data = dataset_test.data
    target = dataset_test.target
    test_data =  DataSet()
    padding = 0
    for i in range(len(data)):
        data_t =  re.sub("\d+|\s+|/", " ", data[i] )
        temp = [word.strip(string.punctuation).lower() for word in data_t.split() if word.strip(string.punctuation) != '']
        test_data.append(Instance(raw = data[i], label = int(target[i]), words = temp))
        if len(temp) > padding:
            padding = len(temp)
    test_data.apply(lambda x: x['raw'].lower(), new_field_name='raw')

    train_data.apply(lambda x: len(x['words']), new_field_name='len')
    test_data.apply(lambda x: len(x['words']), new_field_name='len')

    vocab = Vocabulary(min_freq=10)
    train_data.apply(lambda x: [vocab.add(word) for word in x['words']])
    vocab.build_vocab()
    train_data.apply(lambda x: [vocab.to_index(word) for word in x['words']], new_field_name='seq')
    test_data.apply(lambda x: [vocab.to_index(word) for word in x['words']], new_field_name='seq')
    train_data.rename_field('seq', Const.INPUT)
    train_data.rename_field('len', Const.INPUT_LEN)
    train_data.rename_field('label', Const.TARGET)

    test_data.rename_field('seq', Const.INPUT)
    test_data.rename_field('len', Const.INPUT_LEN)
    test_data.rename_field('label', Const.TARGET)

    test_data.set_input(Const.INPUT, Const.INPUT_LEN)
    test_data.set_target(Const.TARGET)
    train_data.set_input(Const.INPUT, Const.INPUT_LEN)
    train_data.set_target(Const.TARGET)

    test_data, dev_data = test_data.split(0.5)
    return train_data,dev_data,test_data,vocab

class CNN(torch.nn.Module):
    
    def __init__(self, init_embed,
                 num_classes,
                 kernel_nums=(3, 4, 5),
                 kernel_sizes=(3, 4, 5),
                 padding=0,
                 dropout=0.5):
        super(CNN, self).__init__()

        self.embed = encoder.Embedding(init_embed)
        self.conv_pool = encoder.ConvMaxpool(
            in_channels=self.embed.embedding_dim,
            out_channels=kernel_nums,
            kernel_sizes=kernel_sizes,
            padding=padding)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(sum(kernel_nums), num_classes)
    
    def forward(self, words, seq_len=None):
        x = self.embed(words)  # [N,L] -> [N,L,C]
        x = self.conv_pool(x)  # [N,L,C] -> [N,C]
        x = self.dropout(x)
        x = self.fc(x)  # [N,C] -> [N, N_class]
        return {Const.OUTPUT: x}
    
    def predict(self, words, seq_len=None):
        output = self(words, seq_len)
        _, predict = output[Const.OUTPUT].max(dim=1)
        return {Const.OUTPUT: predict}


class RNN(torch.nn.Module):
    
    def __init__(self, embed_num, input_size, hidden_size, target_size):
        super(RNN, self).__init__()
        self.embed = encoder.Embedding((embed_num, input_size))
        self.rnn = encoder.LSTM(  
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=1,  
            batch_first=True, 
        )
        self.output = nn.Linear(hidden_size, target_size)


    def forward(self,  words, seq_len=None):
        x = self.embed(words)
        r_output, (h_n, h_c) = self.rnn(x, None) 
        mean = r_output.sum(1)/r_output.shape[1]
        output = self.output(mean)
        return {Const.OUTPUT: output}


    def predict(self, words, seq_len=None):

        output = self(words, seq_len)
        _, predict = output[Const.OUTPUT].max(dim=1)
        return {Const.OUTPUT: predict}



def init_model():
    train_data,dev_data,test_data,vocab = readdata()



    model = CNN((len(vocab),128), num_classes=target_len, padding=2, dropout=0.1)
    #model = torch.load("rnnmodel/best_RNN_accuracy_2019-05-22-17-18-46")
    trainer = Trainer(model=model, train_data=train_data, dev_data=dev_data, device = 0,
                    save_path='cnnmodel',loss=loss,metrics=metrics,callbacks=[FitlogCallback(test_data)])

    tester = Tester(test_data, model, metrics=AccuracyMetric())
    print(2)
    model2 = RNN(embed_num = len(vocab),input_size = 256, hidden_size  = 256, target_size = target_len)
    #model2 = torch.load("rnnmodel/best_RNN_accuracy_2019-05-22-17-18-46")
    trainer2 = Trainer(model=model2, train_data=train_data, dev_data=dev_data,
                  loss=loss,
                  metrics=metrics,
                  save_path='rnnmodel',
                  batch_size=32,
                  n_epochs=20,
                  device = 0)
    tester2 = Tester(test_data, model, metrics=AccuracyMetric())
    return trainer,trainer2,tester,tester2





def main():
    trainer,trainer2,tester,tester2 = init_model()
    print("finished")
    trainer.train()
    #trainer2.train()
    #tester.test()
    #tester2.test()




if __name__ == '__main__':
    main()

fitlog.finish()