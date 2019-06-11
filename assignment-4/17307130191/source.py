from __init__ import *
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

embedding = 32
poolsize = 2
maxepoch = 10
batch = 16
hidden = 5

train_batch, test_batch, vocabsize = preprocess(batch)

class CNN(nn.Module):
    def __init__(self, embedding, classes=4, padding=1):
        super(CNN, self).__init__()
        wordsize, embeddingsize = embedding
        self.embedding = nn.Embedding(wordsize, embeddingsize)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, padding),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.out = nn.Linear(32 * 15 * 8, classes)

    def forward(self, input):
        embeds = self.embedding(input)
        embeds = embeds.view(batch, 1, -1, embedding)

        out = self.conv1(embeds)
        out = self.conv2(out)
        out = out.view(out.size(0), -1)
        return self.out(out)

class RNN(nn.Module):
    def __init__(self, embedding, classes=4, hidden=5):
        super(RNN, self).__init__()
        wordsize, embeddingsize = embedding
        self.embedding = nn.Embedding(wordsize, embeddingsize)
        self.bh = nn.Parameter(torch.rand(hidden))
        self.bq = nn.Parameter(torch.rand(classes))
        sigma = 40
        self.Wx = nn.Parameter(torch.rand(embeddingsize, hidden) / sigma)
        self.Wh = nn.Parameter(torch.rand(hidden, hidden) /sigma)
        self.Wo = nn.Parameter(torch.rand(hidden, classes) / sigma)
        self.out = nn.Linear(maxlen * classes, classes)

    def forward(self, h_t_1, x_t):
        embeds = self.embedding(x_t)
        h_t = torch.sigmoid(torch.matmul(embeds, self.Wx) + torch.matmul(h_t_1, self.Wh) + self.bh)
        out = torch.matmul(h_t, self.Wo) + self.bq
        return self.out(out.view(out.size(0), -1)), h_t

def train_CNN():
    criterion = nn.CrossEntropyLoss()
    model = CNN((vocabsize, embedding), classes=4, padding=2) #TODO
    optim = torch.optim.RMSprop(model.parameters())
    Loss = []
    for i in range(maxepoch):
        epochloss = torch.tensor(0.)
        for batch_x, batch_y in train_batch: #batch_x is field which is input
            if len(batch_x['words'].numpy()) == batch:
                optim.zero_grad()
                batchloss = torch.tensor(0.)     #batch_y is field which is output
                output = model(torch.LongTensor(batch_x['words']))
                batchloss = criterion(output, batch_y['target']) / batch
                batchloss.backward()
                optim.step()
                epochloss += batchloss
        Loss.append(epochloss)
        print("epoch %d:" %(i), epochloss)
    plt.plot(Loss, label='CNN')
    return model

def train_RNN():
    criterion = nn.CrossEntropyLoss()
    model = RNN((vocabsize, embedding), classes=4, hidden=hidden) #TODO
    optim = torch.optim.RMSprop(model.parameters())
    Loss = []
    for i in range(maxepoch):
        epochloss = torch.tensor(0.)
        h = torch.zeros(batch, maxlen, hidden)
        for batch_x, batch_y in train_batch: #batch_x is field which is input
            if len(batch_x['words'].numpy()) == batch:
                optim.zero_grad()
                batchloss = torch.tensor(0.)     #batch_y is field which is output
                output, h = model(h, torch.LongTensor(batch_x['words']))
                batchloss = criterion(output, batch_y['target']) / batch
                batchloss.backward(retain_graph=True)
                optim.step()
                epochloss += batchloss
        Loss.append(epochloss)
        print("epoch %d:" %(i), epochloss)
    plt.plot(Loss, label='RNN')
    return model

def testCNN(model, batchset):
    sum = 0
    right = 0
    for batch_x, batch_y in batchset:
        if len(batch_x['words'].numpy()) == batch:
            sum += batch
            output = model(torch.LongTensor(batch_x['words']))
            output = torch.max(output, 1)[1]
            for i in range(batch):
                if output[i] == batch_y['target'][i]:
                    right += 1
    return right / sum

def testRNN(model, batchset):
    sum = 0
    right = 0
    h = torch.zeros(batch, maxlen, hidden)
    for batch_x, batch_y in batchset:
        if len(batch_x['words'].numpy()) == batch:
            sum += batch
            output, h = model(h, torch.LongTensor(batch_x['words']))
            output = torch.max(output, 1)[1]
            for i in range(batch):
                if output[i] == batch_y['target'][i]:
                    right += 1
    return right / sum

def main():
    CNNmodel = train_CNN()
    RNNmodel = train_RNN()
    plt.xlabel("Epoch No.")
    plt.ylabel("Loss on training set")
    plt.legend(loc='best')
    plt.show()
    print("on training set:", testCNN(CNNmodel, train_batch))
    print("on test set:", testCNN(CNNmodel, test_batch))
    print("on training set:", testRNN(RNNmodel, train_batch))
    print("on test set:", testRNN(RNNmodel, test_batch))
main()