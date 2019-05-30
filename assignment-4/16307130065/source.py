import os
os.sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
import string
import re
from handout import get_linear_seperatable_2d_2c_dataset
from handout import get_text_classification_datasets
from sklearn.datasets import fetch_20newsgroups
import torch
import torch.nn as nn
from fastNLP import DataSet
from fastNLP import Instance
from fastNLP import Vocabulary
import fastNLP.modules.encoder as encoder
from fastNLP import Trainer, Tester, CrossEntropyLoss, AccuracyMetric


def preprocess(data_in):
    data = data_in.data
    target = data_in.target
    dataset = DataSet()

    for i in range(len(data)):
        data_tmp = re.sub('\d+', ' ', data[i])
        for c in string.whitespace:
            data_tmp = data_tmp.replace(c, ' ')
        for c in string.punctuation:
            data_tmp = data_tmp.replace(c, '')
        data_tmp = data_tmp.lower().split()
        dataset.append(Instance(raw_sentence=data[i], target=int(target[i]), sentence=data_tmp))
    dataset.apply(lambda x: x['raw_sentence'].lower(), new_field_name='raw_sentence')
    dataset.apply(lambda x: len(x['sentence']), new_field_name='seq_len')
    return dataset


dataset_train = fetch_20newsgroups(subset='train', data_home='../../..')
dataset_test = fetch_20newsgroups(subset='test', data_home='../../..')
target_sum = len(dataset_train.target_names)
train_data = preprocess(dataset_train)
test_data = preprocess(dataset_test)

vocab = Vocabulary(min_freq=10).from_dataset(train_data, field_name='sentence')
vocab.index_dataset(train_data, field_name='sentence', new_field_name='words')
vocab.index_dataset(test_data, field_name='sentence', new_field_name='words')

train_data.set_target("target")
train_data.set_input("words")
test_data.set_target("target")
test_data.set_input("words")


# CNN
class CNN(nn.Module):
    def __init__(self, init_embed,
                 num_classes,
                 kernel_nums=(4, 5, 6),
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

    def forward(self, words):
        x = self.embed(words)
        x = self.conv_pool(x)
        x = self.dropout(x)
        x = self.fc(x)
        return {"pred": x}

    def predict(self, words):
        output = self(words)
        _, predict = output["pred"].max(dim=1)
        return {"pred": predict}


trainData, devData = train_data.split(0.2)
model = CNN(init_embed=(len(vocab), 128), num_classes=target_sum, padding=2, dropout=0.1)
trainer = Trainer(model=model, train_data=trainData, dev_data=devData,
                  loss=CrossEntropyLoss(), metrics=AccuracyMetric(),
                  save_path="cnnmodel", batch_size=32, n_epochs=5,
                  device='cuda')
trainer.train()
print('Train finished!')

tester = Tester(test_data, model, metrics=AccuracyMetric())
tester.test()


# RNN
class RNN(nn.Module):
    def __init__(self, init_embed, hidden_dim, num_classes):
        super(RNN, self).__init__()
        self.embed = encoder.Embedding(init_embed)
        self.rnn = encoder.LSTM(
            input_size=self.embed.embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, words):
        x = self.embed(words)
        r_output, _ = self.rnn(x, None)
        x = r_output.sum(1)/r_output.shape[1]
        x = self.fc(x)
        return {'pred': x}

    def predict(self, words):
        output = self(words)
        _, predict = output['pred'].max(dim=1)
        return {'pred': predict}


# trainData, devData = train_data.split(0.2)
# modelRNN = RNN(init_embed=(len(vocab), 128), hidden_dim=128, num_classes=target_sum)
# trainer2 = Trainer(model=modelRNN, train_data=trainData, dev_data=devData,
#                    loss=CrossEntropyLoss(), metrics=AccuracyMetric(),
#                    save_path='rnnmodel', batch_size=32, n_epochs=20,
#                    device='cuda')
# trainer2.train()
# print('Train finished!')
#
# tester = Tester(test_data, modelRNN, metrics=AccuracyMetric())
# tester.test()
