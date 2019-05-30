import torch.nn as nn
from fastNLP import Vocabulary
import torch, time
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn.init as init

class LSTM(nn.Module):
    def init_weight(self, x, y):
        a = torch.zeros(x, y)
        init.xavier_uniform_(a)
        return nn.Parameter(a)

    def init_bias(self, x):
        a = torch.zeros(x)
        return nn.Parameter(a)

    def __init__(self, x_size, h_size, vocab_size):
        super(LSTM, self).__init__()
        self.x_size = x_size
        self.h_size = h_size
        self.out_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, self.x_size)
        self.wf = self.init_weight(x_size + h_size, h_size)
        self.wi = self.init_weight(x_size + h_size, h_size)
        self.wc = self.init_weight(x_size + h_size, h_size)
        self.wo = self.init_weight(x_size + h_size, h_size)
        self.wv = self.init_weight(h_size, vocab_size)

        self.bf = self.init_bias(h_size)
        self.bi = self.init_bias(h_size)
        self.bc = self.init_bias(h_size)
        self.bo = self.init_bias(h_size)
        self.bv = self.init_bias(vocab_size)

    def forward(self, input, h_pre=None, C_pre=None):
        bs, sl = input.size()
        
        if h_pre is None:
            h_pre = torch.zeros(bs, self.h_size).cuda()
        if C_pre is None:
            C_pre = torch.zeros(bs, self.h_size).cuda()
        
        out_prob = torch.zeros(sl, bs, self.out_size).cuda()
        # input [sl, bs] x [sl, bs, x_size]
        x = self.embedding(input.transpose(0, 1))
        
        
        for i in range(sl):
            z = torch.cat((x[i, :, :], h_pre), dim=1)
            ft = torch.sigmoid(z.matmul(self.wf) + self.bf)
            it = torch.sigmoid(z.matmul(self.wi) + self.bi)
            bar_Ct = torch.tanh(z.matmul(self.wc) + self.bc)
            Ct = ft * C_pre + it * bar_Ct
            ot = torch.sigmoid(z.matmul(self.wo) + self.bo)
            ht = ot * torch.tanh(Ct)
            out_prob[i, :, :] = ht.matmul(self.wv) + self.bv
            C_pre = Ct.clone()
            h_pre = ht.clone()

        return out_prob, h_pre, C_pre

class Dataset(object):
    def __init__(self, shi):
        self.shi = shi
        self.num = shi.shape[0]
        
    def __getitem__(self, item):
        x = self.shi[item, :]
        label = torch.zeros(x.shape)
        label[:-1], label[-1] = x[1:], x[0]
        return x, label
    
    def __len__(self):
        return self.num

def get_data(path):
    f = open(path, 'r')
    shi = f.read()
    shi = shi.replace('\n', '').replace('\r', '')
    shi = shi[:5000 * 64]
    sl = 64
    l_doc = len(shi)
    vocab = Vocabulary(min_freq=1)
    for i in shi:
        vocab.add(i)
    vocab.build_vocab()
    vocab_size = len(vocab)
    num_s = int(l_doc / sl)
    train_s = int(num_s * 0.8)
    test_s = num_s - train_s
    array_shi = torch.zeros(l_doc)
    train_shi = torch.zeros(train_s, sl)
    test_shi = torch.zeros(test_s, sl)
    print(train_shi.size())
    print(test_shi.size())
    for i, j in enumerate(shi):
        array_shi[i] = vocab[j]

    array_shi = array_shi.view(-1, sl)
    train_shi[:, :] = array_shi[:train_s, :]
    test_shi[:, :] = array_shi[train_s:, :]

    return vocab, train_shi, test_shi

def calc_perplexity(net, test_set):
    per, cnt = 0, 0
    for x, label in test_set:
        x = x.cuda()
        label = label.cuda()
        output_prob, __, ___ = net(x.long())
        output_prob = output_prob.permute(1, 0, 2).contiguous()
        output_prob = F.softmax(output_prob, dim=2)
        bs, sl = label.size(0), label.size(1)
        for i in range(bs):
            p = 1
            for j in range(sl - 1):
                real_one = int(label[i][j].item())
                p = p * np.power(1.0 / output_prob[i][j][real_one].cpu().detach().numpy(), 1.0 / sl)
            per += p

        cnt += bs

    return per / cnt

def choose(prob):
    __, label = torch.topk(prob, 3, 0)
    a = np.random.randint(0, 3)
    return label[a].item()


def output(net, vocab, vocab_size, be):
    sl = 64
    net = net.eval()
    out_res = be
    x = vocab[be]
    input = torch.LongTensor(1)
    input[0] = x
    input = input.view(1, 1)
    output_prob, h, c = net(input.cuda())
    for i in range(sl - 1):
        output_prob, h, c = net(input.cuda(), h, c)
        output_prob = output_prob.permute(1, 0, 2).contiguous()
        output_prob = output_prob.view(vocab_size, 1)
        x = choose(output_prob)
        out_res += vocab.to_word(x)
        input[0] = x
        input = input.view(1, 1)

    print(out_res)

def engine(net, train_set, test_set, vocab, vocab_size):
    CELoss = nn.CrossEntropyLoss().cuda()
    net = net.cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.003)
    per_list, loss_list = [], []
    epoch = 0
    tf = time.time()
    while True:
        loss_sum = 0
        cnt = 0
        for x, label in train_set:
            x = x.cuda()
            label = label.cuda()
            output_prob, __, ___ = net(x.long())
            output_prob = output_prob.permute(1, 0, 2).contiguous()
            output_prob = output_prob.view(-1, vocab_size)
            label = label.view(-1)
            loss = CELoss(output_prob, label.long())
            loss_val = loss.item()
        
            net.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cnt += 1
            loss_sum += loss_val

        epoch += 1
        per = calc_perplexity(net, test_set)
        interval = time.time() - tf
        tf = time.time()
        loss_avg = loss_sum / cnt
        per_list.append(per)
        loss_list.append(loss)
        print("Time:{:.2f}\tEpoch:{:d}\tLoss:{:.2f}\tPerplexity:{:.2f}".format(interval, epoch, loss_sum / cnt, per))

        if epoch > 10:
            per_avg = 0
            for i in range(epoch - 11, epoch - 1):
                per_avg += per_list[i]
            per_avg /= 10
            if per_avg <= per:
                break

        if epoch >= 150:
            break

    loss_np = np.array(loss_list)
    #np.save("sgd.npy", loss_np)


    #output(net, vocab, vocab_size, "日")
    #output(net, vocab, vocab_size, "红")
    #output(net, vocab, vocab_size, "山")
    #output(net, vocab, vocab_size, "夜")
    #output(net, vocab, vocab_size, "湖")
    #output(net, vocab, vocab_size, "海")
    #output(net, vocab, vocab_size, "月")
        
   


if __name__ == "__main__":
    batch_size = 64
    x_size = 128
    h_size = 256

    path = "poetry.txt"
    vocab, train_shi, test_shi = get_data(path)
    train_data = Dataset(train_shi)
    test_data = Dataset(test_shi)
    vocab_size = len(vocab)
    print(vocab_size)
    lstm = LSTM(x_size, h_size, vocab_size)
    train_set = DataLoader(train_data, batch_size, True, num_workers=4)
    test_set = DataLoader(test_data, batch_size, False, num_workers=4)
    engine(lstm, train_set, test_set, vocab, vocab_size)






