import os
import torch
import torch.nn as nn
import torch.optim as optim
import data_processing
from model_batched import LSTM_torch


def train(epochNum=50, batch=128, lr=0.01, embed_size=256, hidden_size=256):
    D = data_processing.Dataset_batched_padding()
    src = './model_padding_10k/'  # 存放模型
    if not os.path.exists(src):
        os.mkdir(src)
    model = LSTM_torch(len(D.word_to_ix), embed_size, hidden_size)
    # model = torch.load(src+'poetry-gen-epoch20-loss4.082079.pt') # reload model, for future training
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
    criterion = nn.NLLLoss()  # together with logsoftmax, form CrossEntrophyloss

    TRAINSIZE = len(D.train)
    TESTSIZE = len(D.develop)

    def test():
        model.eval()
        loss = 0
        perplexity = 0
        for batchIndex in range(int(TESTSIZE / batch)):
            if batchIndex * batch >= TESTSIZE:
                continue
            t, o = D.makeForOneBatch(D.develop[batchIndex * batch:min((batchIndex + 1) * batch, TESTSIZE)])
            output, hidden = model(t)
            for i in range(o.size()[0]):
                loss += criterion(output[i, :, :], o[i, :])
                perplexity += compute_perplexity(output[i, :, :], o[i, :])

        loss = loss / TESTSIZE
        perplexity = perplexity / TESTSIZE
        print("=====", loss.item())
        print("+++++", perplexity.item())
        return loss.item()


    print("start training")
    for epoch in range(epochNum):
        model.train()
        for batchIndex in range(int(TRAINSIZE / batch)):
            if batchIndex * batch >= TRAINSIZE:
                continue
            model.zero_grad()
            t, o = D.makeForOneBatch(D.train[batchIndex * batch:min((batchIndex + 1) * batch, TRAINSIZE)])
            output, hidden = model(t)
            loss = 0
            for i in range(o.size()[0]):
                loss += criterion(output[i, :, :], o[i, :])
            loss = loss / o.size()[0]
            loss.backward()
            print(epoch, loss.item())
            optimizer.step()

        test_loss = test()
        torch.save(model, src + 'poetry-gen-epoch%d-loss%f.pt' % (epoch + 1, test_loss))


def compute_perplexity(output, target):  # 计算每个句子的困惑度
    N = len(target)
    return 2**(-1/N*sum([output[j][target[j]] for j in range(len(target))]))

