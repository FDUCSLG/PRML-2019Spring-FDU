from fastNLP import BucketSampler
from fastNLP import Batch
import torch
import torch.nn as nn


def cnn_train(epoch, data, model, batch_size = 32):
    device = torch.device("cuda")
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    lossfunc = nn.CrossEntropyLoss()

    train_sampler = BucketSampler(batch_size=batch_size, seq_len_field_name='seq_len')
    train_batch = Batch(batch_size=batch_size, dataset=data, sampler=train_sampler)

    for i in range(epoch):
        loss_list = []
        cnt = 0
        for batch_x, batch_y in train_batch:
            optim.zero_grad()
            batch_x['words'] = batch_x['words'].long().contiguous().to(device)
            batch_y['target'] = batch_y['target'].long().contiguous().to(device)
            output = model(batch_x['words'])
            loss = lossfunc(output['pred'], batch_y['target'])
            loss.backward()
            optim.step()
            loss_list.append(loss.item())
            info_str = '[info] Epoch {:d} Iteration {:d} Loss : {:.2f}'.format(i, cnt, loss_list[-1])
            print(info_str)
            with open('./cnn_rec.txt', 'a') as fp:
                fp.write(info_str)
                fp.write('\n')
            cnt += 1
        loss_list.clear()
        torch.save(model.state_dict(), './cnn_state.pth')

def rnn_train(epoch, data, model, batch_size = 32):
    device = torch.device("cuda")
    optim = torch.optim.Adam(model.parameters(), lr=0.002)
    lossfunc = nn.CrossEntropyLoss()

    train_sampler = BucketSampler(batch_size=batch_size,seq_len_field_name='seq_len')
    train_batch = Batch(batch_size=batch_size, dataset=data, sampler=train_sampler)

    for i in range(epoch):
        loss_list = []
        cnt = 0
        for batch_x, batch_y in train_batch:
            optim.zero_grad()
            batch_x['words'] = batch_x['words'].long().contiguous().to(device)
            batch_y['target'] = batch_y['target'].long().contiguous().to(device)
            output = model(batch_x['words'])
            loss = lossfunc(output['pred'], batch_y['target'])
            loss.backward()
            optim.step()
            loss_list.append(loss.item())
            info_str = '[info] Epoch {:d} Iteration {:d} Loss : {:.2f}'.format(i, cnt, loss_list[-1])
            print(info_str)
            with open('./rnn_rec.txt', 'a') as fp:
                fp.write(info_str)
                fp.write('\n')
            cnt += 1
        loss_list.clear()
        torch.save(model.state_dict(), './rnn_state.pth')
