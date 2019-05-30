import collections
import time
import math
import pickle
import torch
import torch.nn as nn
import torch.optim as optim

from create_dataset import *
from data_process import *
from rnn import RNN


gitclone()

poetry_list = create_POETRY_LIST()
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

print(n_words)

pickle.dump({'word_list' : word_list}, open('./word_list', 'wb'))


rnn = RNN(n_words, 256, 256)
rnn.cuda()

# optimizer = optim.RMSprop(rnn.parameters(), lr=0.01, weight_decay=0.0001)
# optimizer = optim.SGD(rnn.parameters(), lr=10.0)
optimizer = optim.Adam(rnn.parameters(), lr=0.01)
criterion = nn.NLLLoss()

n_epochs = 10
batch = 200
n_batches = n_poetries // batch
n_batches = 50
n_batches_for_train = int(n_batches * 0.8)
n_batches_for_develop = n_batches - n_batches_for_train
pre_perplexity = None
perplexity = 0.0
start = time.time()

def develop():
    loss = 0
    for i_poetry in range(n_batches_for_train * batch, (n_batches_for_train + 1) * batch):
        poetry = poetry_list[i_poetry]
        input, target = create_training_data(poetry, word_list)
        hidden = rnn.init_hidden()
        try:
            output, hidden = rnn(input.cuda(), hidden)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory')
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                print('| WARNING: ran out of memory, skipping batch')
                break
            else:
                raise e
        loss += criterion(output, target.cuda())
    loss = loss / (batch)
    print('develop: %.5f' % (loss.item()))
    return loss.item()

print('train begin')
for epoch in range(n_epochs):
    for i_batch in range(n_batches_for_train):
        rnn.zero_grad()
        loss = 0
        flag = False
        for i_poetry in range(i_batch * batch, (i_batch + 1) * batch):
            poetry = poetry_list[i_poetry]
            input, target = create_training_data(poetry, word_list)
            hidden = rnn.init_hidden()

            try:
                output, hidden = rnn(input.cuda(), hidden)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory')
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    print('| WARNING: ran out of memory, skipping batch')
                    flag = True
                    break
                else:
                    raise e

            loss += criterion(output, target.cuda())
        if flag:
            continue
        loss = loss / batch

        try:
            loss.backward()
            optimizer.step()
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory')
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                print('| WARNING: ran out of memory, skipping batch')
                continue
            else:
                raise e

        print('train: %d %d %.5f' % (epoch, i_batch, loss.item()))

    develop_loss = develop()
    perplexity = math.exp(develop_loss)
    print('perplexity: %d %.5f' % (epoch, perplexity))
    torch.save(rnn, './rnn_%d_%.5f_Adam.pt' % (epoch, perplexity))

    end = time.time()
    print('%.5f s' % (end - start))
    start = end