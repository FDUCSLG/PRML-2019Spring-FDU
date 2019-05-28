import os
os.sys.path.append('../../assignment-2')
from handout import get_text_classification_datasets
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
import matplotlib.pyplot as plt
import numpy as np
from fastNLP import DataSet
from fastNLP import Instance
from fastNLP import Vocabulary
from fastNLP import BucketSampler
from fastNLP import Batch
import re
import string

embedding = 300
batch = 16
vocabsize = 0
maxepoch = 60

traindata, testdata = get_text_classification_datasets()

def pre(x):
    data = re.sub(r'[^a-zA-Z0-9\s]','',x)
    data = re.sub(r'['+string.whitespace+']+',' ',data)
    return re.split(r' +', data.strip().lower())

def tobatch(dataset):
    size = (len(dataset) - 1) // batch
    X = []
    for i in range(size):
        slice = dataset[i * batch : (i + 1) * batch]
        maxlen = 0
        maxlen = [max(maxlen, len(sentence)) for sentence in slice]
        temp = np.full((batch, maxlen), vocab.to_index(" "), np.int32)
        for j in range(batch):
            temp[j][maxlen - len(slice):] = slice[j]
        X.append(temp)
    return X


def preprocess():
    raw_data1 = []
    raw_data2 = []

    for i in range(len(traindata.data)):
        raw_data1.append(Instance(sentence=traindata.data[i], label=int(traindata.target[i])))
    trainset = DataSet(raw_data1)
    trainset.apply(lambda x: pre(x['sentence']), new_field_name='words', is_input=True)

    for i in range(len(testdata.data)):
        raw_data2.append(Instance(sentence=testdata.data[i], label=int(testdata.target[i])))
    testset = DataSet(raw_data2)
    testset.apply(lambda x: pre(x['sentence']), new_field_name='words')
    global vocab
    vocab = Vocabulary(min_freq=1).from_dataset(trainset, testset, field_name='words')
    vocab.index_dataset(trainset, testset, field_name='words', new_field_name='words')

    trainset.apply(lambda x: int(x['label']), new_field_name='target', is_target=True)
    testset.apply(lambda x: int(x['label']), new_field_name='target', is_target=True)

    vocab.add_word(" ")
    global  vocabsize
    vocabsize = len(vocab)
    sampler = BucketSampler(batch_size=batch, seq_len_field_name='seq_len')
    train_batch = Batch(batch_size=batch, dataset=trainset, sampler=sampler)
    test_batch = Batch(batch_size=batch, dataset=testset, sampler=sampler)

    return train_batch, test_batch

class CNN(nn.Module):
    def __init__(self, embedding, classes=4, padding=1, dropout=0):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv
        )
        wordsize, embeddingsize = embedding
        self.embedding = nn.embedding(wordsize, embeddingsize)


    def foward(self, input):
        embeds = self.embedding(input).t()

def train():
    train_batch, test_batch = preprocess()
    criter
    optim = torch.optim.PMSprop()
    Loss = []
    for i in range(maxepoch):
        epochloss = torch.tensor(0.)
        for batch_data in train_batch:
            print(batch_data['words'])
            #input = torch.tensor([train_batch[i]["words"][0]])
            #for j in range(1, len(train_batch[i]["words"])):
            #    input = torch.cat((input, torch.tensor([train_batch[i]["words"][j]])), 0)


        # model = CNN(vocabsize, embedding), classes=4, padding=2, dropout=0.1)

train()