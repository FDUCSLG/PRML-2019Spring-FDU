from fastNLP import DataSet
from fastNLP import Vocabulary
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np
import os
import json
import re
import math

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.ouput_size = output_size
        m, n, dim = output_size, hidden_layer_size, input_size
        sigma = 20
        self.embedding = nn.Embedding(m, dim)
        self.W_A = nn.Parameter(torch.rand(m, n, dtype=torch.float32) / sigma)
        self.W_C = nn.Parameter(torch.rand(n, n + dim, dtype=torch.float32) / sigma)
        self.W_f = nn.Parameter(torch.rand(n, n + dim, dtype=torch.float32) / sigma)
        self.W_o = nn.Parameter(torch.rand(n, n + dim, dtype=torch.float32) / sigma)
        self.W_i = nn.Parameter(torch.rand(n, n + dim, dtype=torch.float32) / sigma)
        self.b_A = nn.Parameter(torch.rand(m, 1, dtype=torch.float32))
        self.b_C = nn.Parameter(torch.rand(n, 1, dtype=torch.float32))
        self.b_f = nn.Parameter(torch.rand(n, 1, dtype=torch.float32))
        self.b_o = nn.Parameter(torch.rand(n, 1, dtype=torch.float32))
        self.b_i = nn.Parameter(torch.rand(n, 1, dtype=torch.float32))

    def forward(self, input_text, h_prev, C_prev):
        input = self.embedding(input_text).t()
        z = torch.cat((h_prev, input), 0)
        f = torch.sigmoid(self.W_f.mm(z) + self.b_f)
        i = torch.sigmoid(self.W_i.mm(z) + self.b_i)
        o = torch.sigmoid(self.W_o.mm(z) + self.b_o)
        tildeC = torch.tanh(self.W_C.mm(z) + self.b_C)
        C = f * C_prev + i * tildeC
        h = o * torch.tanh(C)
        y = self.W_A.mm(h) + self.b_A
        return y, h, C


def perplexity():
    avg = 0
    for num in range(0, len(test_data)):
        h, c = torch.zeros(n, 1), torch.zeros(n, 1)
        for t in range(0, sl):
            inputs = torch.tensor([vocab[test_data[num][t]]])
            outputs, h, c = net(inputs, h, c)
            if t < sl - 1:
                labels = torch.tensor([vocab[test_data[num][t + 1]]])
            else:
                labels = torch.tensor([vocab["EOS"]])
            loss_t = loss(outputs.t(), labels)
            avg += loss_t.item()
    avg = avg / len(test_data) / sl
    return avg


def Train(batch_size, f):
    lst_perp, epoch = perplexity(), 0
    print("Perplexity of Epoch %d is %.5f" % (epoch, lst_perp))
    train_size = len(train_data)
    train_Perp_Record = [torch.tensor([lst_perp])]
    Perp_Record = [lst_perp]
    while True:
        try:
            epoch += 1
            train_loss = 0
            for num in range(0, train_size, batch_size):
                size = min(train_size - num, batch_size)
                optimizer.zero_grad()
                Loss = torch.tensor(0.)
                h, c = torch.zeros(n, size), torch.zeros(n, size)
                for t in range(0, sl):
                    inputs = torch.tensor([vocab[train_data[num][t]]])
                    for j in range(num + 1, num + size):
                        inputs = torch.cat((inputs, torch.tensor([vocab[train_data[j][t]]])), 0)
                    outputs, h, c = net(inputs, h, c)
                    if t < sl - 1:
                        labels = torch.tensor([vocab[train_data[num][t+1]]])
                        for j in range(num + 1, num + size):
                            labels = torch.cat((labels, torch.tensor([vocab[train_data[j][t+1]]])), 0)
                    else:
                        labels = torch.tensor([vocab["EOS"]])
                        for j in range(num + 1, num + size):
                            labels = torch.cat((labels, torch.tensor([vocab["EOS"]])), 0)
                    loss_t = loss(outputs.t(), labels)
                    Loss += loss_t

                Loss /= sl
                train_loss += Loss * size
                Loss.backward()
                optimizer.step()
            train_loss /= train_size
            #now_perp = perplexity(data=train_data)
            print("Perplexity of Epoch %d on training_set is %.5f" % (epoch, train_loss))
            train_Perp_Record.append(train_loss)
            if epoch >= 75: break
            # now_perp = perplexity()
            # print("Perplexity of Epoch %d is %.5f" % (epoch, now_perp))
            # Perp_Record.append(now_perp)
            # if now_perp > lst_perp + 1e-2: break
            # lst_perp = now_perp
        except KeyboardInterrupt:
            break
    plt.xlabel("Epoch No.")
    plt.ylabel("Loss Function on test set")
    plt.plot(np.linspace(0, len(Perp_Record), len(Perp_Record)), Perp_Record)
    plt.show()

    plt.xlabel("Epoch No.")
    plt.ylabel("Loss Function on training set")
    plt.plot(np.linspace(0, len(train_Perp_Record), len(train_Perp_Record)), train_Perp_Record)
    plt.show()
    for i in train_Perp_Record:
        f.write(str(i.item()) + "\n")


def Output(word):
    fout = open("%s.txt" % word, "w")
    fout.write(word)
    inputs = torch.tensor([vocab[word]])
    h, c = torch.zeros(n, 1), torch.zeros(n, 1)
    for t in range(0, sl):
        outputs, h, c = net(inputs, h, c)
        outputs = torch.log_softmax(outputs, 0)
        new_word = vocab.to_word(np.random.choice(m, 1, p=np.exp(outputs.view(-1).data.numpy()))[0])
        fout.write(new_word)
        inputs = torch.tensor([vocab[new_word]])
    print("%s Complete!" % word)


if __name__ == '__main__':
    filedir = os.getcwd() + "/dataset"
    filenames = os.listdir(filedir)
    data = []
    for i in filenames:
        filepath = filedir + '/' + i
        f = open(filepath, 'rb')
        dat = json.load(f)
        data.append(dat)
    fil = re.compile(r'[\\s+\\.\\!\\/_,$%^*(+\\\"\')]+ | [+——()?【】“”！，。？、~ @  # ￥%……&*（）]+')
    poems = []
    for i in data:
        for j in i:
            tmp = ""
            for k in j['paragraphs']:
                tmp += k
            tmp = re.sub(fil, '', tmp)
            if len(tmp) < 20: continue
            elif len(tmp) < 60: tmp = tmp.rjust(60)
            poems.append(tmp[:60])
    print(len(poems))
    poems = poems[:100]
    sep = math.ceil(len(poems) * 0.8)
    train_data = poems[:sep]
    test_data = poems[sep:]

    # Preprocess
    sl = 60
    # dataSet = DataSet.read_csv("../handout/tangshi.txt", headers=["raw_sentence"])
    # dataSet.drop(lambda x: len(x['raw_sentence']) != sl)
    # test_data, train_data = dataSet.split(0.8)

    vocab = Vocabulary(min_freq=1)
    # dataSet.apply(lambda x: [vocab.add(word) for word in x['raw_sentence']])
    for i in poems:
        for word in i:
            vocab.add(word)
    vocab.add("EOS")
    vocab.build_vocab()
    print(len(vocab))
    m, n, dim = len(vocab), 64, 64
    net = LSTM(dim, n, m)
    #net.load_state_dict(torch.load('state1'))
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    Train(batch_size=1, f=open('Momentum', 'w'))
    # torch.save(net.state_dict(), "state2")

    # Output
    Output("日")
    Output("红")
    Output("山")
    Output("夜")
    Output("湖")
    Output("海")
    Output("月")
