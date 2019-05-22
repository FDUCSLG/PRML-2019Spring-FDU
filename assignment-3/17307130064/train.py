import collections
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim

from data_process import *
from rnn import RNN


poetry_list = []
file_path = './../handout/tangshi.txt'
file = open(file_path, encoding='utf-8')
poetries = file.read().strip().split('\n\n')
poetry_list = [poetry.replace('\n', '') for poetry in poetries]
n_poetries = len(poetry_list)

word_list = []
for poetry in poetry_list:
    word_list.extend([word for word in poetry])
counter = collections.Counter(word_list)
sorted_word_list = sorted(counter.items(), key=lambda x : x[1], reverse=True)
word_list = [x[0] for x in sorted_word_list]
word_list.append('E')
n_words = len(word_list)
poetry_list = [poetry + 'E' for poetry in poetry_list]


def develop():
    loss = 0
    for i_batch in range(n_batches_for_train, n_batches_for_train + n_batches_for_develop):
        for i_poetry in range(i_batch * batch, (i_batch + 1) * batch):
            poetry = poetry_list[i_poetry]
            input, target = create_training_data(poetry, word_list)
            hidden = rnn.init_hidden()
            output, hidden = rnn(input, hidden)
            loss += criterion(output, target)
    loss = loss / (n_batches_for_develop * batch)
    print('develop: %.5f' % (loss))
    return loss

rnn = RNN(n_words, 256, 256)

optimizer = optim.RMSprop(rnn.parameters(), lr=0.01, weight_decay=0.0001)
# optimizer = optim.SGD(rnn.parameters(), lr=10.0, momentum=0.9)
# optimizer = optim.Adam(rnn.parameters(), lr=0.1)
criterion = nn.NLLLoss()

n_epochs = 10
batch = 30
n_batches = n_poetries // batch
n_batches_for_train = int(n_batches * 0.8)
n_batches_for_develop = n_batches - n_batches_for_train
pre_perplexity = None
perplexity = 0.0
start = time.time()

for epoch in range(n_epochs):
    for i_batch in range(n_batches_for_train):
        rnn.zero_grad()
        loss = 0
        for i_poetry in range(i_batch * batch, (i_batch + 1) * batch):
            poetry = poetry_list[i_poetry]
            input, target = create_training_data(poetry, word_list)
            hidden = rnn.init_hidden()
            output, hidden = rnn(input, hidden)
            loss += criterion(output, target)
        loss = loss / batch
        loss.backward()
        optimizer.step()
        print('train: %d %d %.5f' % (epoch, i_batch, loss.item()))

        develop_loss = develop()
        perplexity = math.exp(develop_loss)
        print('perplexity: %d %d %.5f' % (epoch, i_batch, perplexity))

        torch.save(rnn, './rnn_%d_%d_%.5f_RMSprop.pt' % (epoch, i_batch, perplexity))

        end = time.time()
        print('%.5f s' % (end - start))
        start = end