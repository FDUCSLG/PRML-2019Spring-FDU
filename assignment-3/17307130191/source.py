import sys
import json
import torch
from torch.autograd import Variable 
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

sys.path.append("../")

class Config(object):
    max_epoch = 30
    batch_size = 1
    embedding_dim = 64 
    hidden_dim = 64    
    save_every = 4

def preprocess(frac, conf):
    data_path = "./tangshi.json"
    raw_poems = []
    poems = []
    with open(data_path, "r", encoding="UTF-8") as f:
        all = json.loads(f.read())
        num = 0
        for line in all:
            poem = line['paragraphs']
            if len(poem) == 2:
                num += 1
                raw_poems.append(poem)

    allwords = {}
    for poem in raw_poems:
        poems.append(''.join(poem) + "$")

    for poem in poems:  
        for word in poem:
            if word not in allwords:
                allwords[word] = 1
            else:
                allwords[word] += 1
    wordpairs = sorted(allwords.items(), key = lambda x : -x[1])
    words, a = zip(*wordpairs)
    words += " ",
    word2id = dict(zip(words, range(len(words))))
    wordindex = lambda A: word2id.get(A, len(words))
    poems2vec = [([wordindex(word) for word in poem]) for poem in poems]
    
    train_data = []
    test_data = []
    trainnum = int(len(poems2vec) * frac)
    trainnum = len(poems2vec) - trainnum
    train_data = poems2vec[:trainnum]
    test_data = poems2vec[trainnum:]
    X_t = batch(train_data, wordindex, word2id, conf)
    X_d = batch(test_data, wordindex, word2id, conf)
    return X_t, X_d, word2id, wordindex

def batch(data, wordindex, word2id, conf):
    batchsize = conf.batch_size
    
    batchnum = (len(data) - 1) // batchsize
    X = []
    for i in range(batchnum):
        maxlen = 0
        batch = data[i * batchsize: (i + 1) * batchsize]
        for j in range(batchsize):
            maxlen = max(maxlen, len(batch[j]))

        temp = np.full((batchsize, maxlen), wordindex(" "), np.int32)
        for j in range(batchsize):
            temp[j][(maxlen - len(batch[j])):] = batch[j]
        X.append(temp)
    return X
      
class Poemmodel(nn.Module):
    def __init__(self, conf, word2id): #TODO
        super(Poemmodel, self).__init__()
        self.input_dim = conf.embedding_dim
        self.output_dim = conf.hidden_dim
        self.word2id = word2id
        self.conf = conf
        sigma = 20
        raw_input_dim = len(word2id)
        self.bi=nn.Parameter(torch.rand(self.output_dim, 1))
        self.bf=nn.Parameter(torch.rand(self.output_dim, 1))
        self.bc=nn.Parameter(torch.rand(self.output_dim, 1))
        self.bo=nn.Parameter(torch.rand(self.output_dim, 1))
        self.ba=nn.Parameter(torch.rand(raw_input_dim, 1))
        
        self.Wa=nn.Parameter(torch.rand(raw_input_dim, self.input_dim) / sigma)
        self.Wi=nn.Parameter(torch.rand(self.output_dim, self.output_dim + self.input_dim) / sigma)
        self.Wf=nn.Parameter(torch.rand(self.output_dim, self.output_dim + self.input_dim) / sigma)
        self.Wc=nn.Parameter(torch.rand(self.output_dim, self.output_dim + self.input_dim) / sigma)
        self.Wo=nn.Parameter(torch.rand(self.output_dim, self.output_dim + self.input_dim) / sigma)

        self.embedding = nn.Embedding(raw_input_dim, self.input_dim)

    def forward(self, input, hidden=None):

        embeds = self.embedding(input).t() # [300 * 1900] * [1900, n]
        (h_t_1, c_t_1) = hidden # 300 * n

        # print("embeds:", len(embeds), len(embeds[0]))
        z = torch.cat((h_t_1, embeds), 0)
        i_t = torch.sigmoid(self.Wi.mm(z) + self.bi)
        f_t = torch.sigmoid(self.Wf.mm(z) + self.bf)
        o_t = torch.sigmoid(self.Wo.mm(z) + self.bo)
        c_o_t = torch.tanh(self.Wc.mm(z) + self.bc)
        c_t = f_t * c_t_1 + i_t * c_o_t
        h_t = o_t * torch.tanh(c_t)
        output = self.Wa.mm(h_t) + self.ba # [1900 * 300] * [300 * n]
        return output, h_t, c_t
    
def train():
    conf = Config()
    train_data, dev_data, word2id, wordindex = preprocess(0.2, conf)
    model = Poemmodel(conf, word2id)
    optimizer = torch.optim.RMSprop(model.parameters())
    len_train_data = len(train_data)

    # net_SGD = Poemmodel(conf, word2id)
    # net_Momentum = Poemmodel(conf, word2id)
    # net_RMSprop = Poemmodel(conf, word2id)
    # net_Adam = Poemmodel(conf, word2id)
    # nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]

    # opt_SGD = torch.optim.SGD(net_SGD.parameters(), lr=0.01)
    # opt_Momentum = torch.optim.SGD(net_Momentum.parameters(), lr=0.01, momentum=0.9)
    # opt_RMSprop = torch.optim.RMSprop(net_RMSprop.parameters())
    # opt_Adam = torch.optim.Adam(net_Adam.parameters())
    # optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]
    # losses = [[],[],[],[]]
    criterion = nn.CrossEntropyLoss()
    # model.load_state_dict(torch.load('model.path'))
    Loss = []
    for i in range(conf.max_epoch):
        loss1 = torch.tensor(0.)
        for poems in train_data:
            loss3 = torch.tensor(0.)
            optimizer.zero_grad()
            lenth = len(poems[0])
            for j in range(conf.batch_size):#TODO
                h = torch.zeros(model.output_dim, lenth - 1)
                c = torch.zeros(model.output_dim, lenth - 1) # 64 * n
                input = torch.LongTensor([poems[j][0]])
                for k in range(1, lenth - 1):
                    input = torch.cat((input, torch.LongTensor([poems[j][k]])), 0)
                output, h, c = model(input, (h, c))
                tar = torch.LongTensor([poems[j][1]])
                for k in range(1, lenth - 1):
                    tar = torch.cat((tar, torch.LongTensor([poems[j][k + 1]])), 0)
                loss3 += criterion(output.t(), tar)
                # print(input, tar)
            loss3.backward()
            optimizer.step()
            loss1 += loss3 / (lenth - 1)
        loss2 = loss1 / (conf.batch_size * len(train_data))
        Loss.append(loss2)
        print("epoch loss:", loss2)
        if i % conf.save_every == 0 and i != 0:
            torch.save(model.state_dict(), 'model_epoch_%s.path'%(i))
    perplexity(model, dev_data)
    # labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']
    # for i, l_his in enumerate(losses):
    #     plt.plot(l_his, label=labels[i])
    # plt.legend(loc='best')
    plt.xlabel("Epoch No.")
    plt.ylabel("Loss function on training set")
    plt.plot(np.linspace(0, len(Loss), len(Loss)), Loss)
    plt.show()
    generate(wordindex("日"), model, wordindex)
    # generate(wordindex("红"), model, wordindex)
    generate(wordindex("山"), model, wordindex)
    generate(wordindex("夜"), model, wordindex)
    generate(wordindex("湖"), model, wordindex)
    generate(wordindex("海"), model, wordindex)
    generate(wordindex("月"), model, wordindex)
    generate(wordindex("红"), model, wordindex)

def generate(prefix, model, wordindex):
    next = prefix
    vec = []
    vec.append(prefix)
    h = torch.zeros(model.output_dim, 1)
    c = torch.zeros(model.output_dim, 1)
    output, h, c = model(torch.LongTensor([next]), (h, c))
    output = torch.max(output, 0)[1]
    while output != torch.LongTensor([wordindex("$")]):
        # print([k for k, v in model.word2id.items() if v == output.data.numpy()])
        vec.append(output.data.numpy())
        output, h, c = model(output, (h, c))
        output = torch.max(output, 0)[1]
    set = ""
    for i in range(len(vec)):
        set += ''.join([k for k, v in model.word2id.items() if v == vec[i]])
    print(set)

def perplexity(model, dev_data):
    criterion = nn.CrossEntropyLoss()
    loss1 = torch.tensor(0.)
    for poems in dev_data:
        loss3 = torch.tensor(0.)
        lenth = len(poems[0])
        for j in range(model.conf.batch_size):#TODO
            h = torch.zeros(model.output_dim, 1)
            c = torch.zeros(model.output_dim, 1)
            for k in range(lenth - 1):
                input = poems[j][k]
                output, h, c = model(torch.LongTensor([input]), (h, c))
                tar = torch.LongTensor([poems[j][k + 1]])
                loss3 += criterion(output.view(1, -1), tar)
        loss1 += loss3 / (lenth - 1)
    print("perplexity:", loss1 / (len(dev_data) * model.conf.batch_size))
train()

