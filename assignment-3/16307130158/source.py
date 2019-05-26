import re
import numpy as np
from fastNLP import Vocabulary
import torch
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim

# 全唐诗预处理
def data_process():
    with open('./data.txt', encoding='utf-8') as fp:
        out = fp.readlines()
        data = list(out)

    poem = []
    cnt = 0
    for temp in data:
        cnt += 1
        if cnt % 2 == 0:
            rec = re.sub('，', '', temp)
            poem.append(rec[:-1])

    poem_normalized = []
    for i in range(len(poem)):
        if len(poem[i]) < 80:
            poem[i] = ' '*(80 - len(poem[i])) + poem[i]
            poem_normalized.append(poem[i])
        else:
            poem_normalized.append(poem[i][:80])

    vocab = Vocabulary(min_freq=2)
    for temp in poem_normalized:
        for x in temp:
            vocab.add(x)

    vocab.build_vocab()
    dataset = []
    for temp in poem_normalized:
        dataset.append([vocab.to_index(x) for x in temp])
    return vocab, np.array(dataset)

# 超参数设置
class Config(object):
    lr = 1e-2
    epoch = 20
    batch_size = 64
    max_gen_len = 28
    embedding_dim = 128
    hidden_dim = 128

# LSTM 模型
class LSTM_model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # input gate
        self.wxi = nn.Parameter(torch.rand(embedding_dim, hidden_dim) * np.sqrt(2 / (embedding_dim + hidden_dim)))
        self.whi = nn.Parameter(torch.rand(hidden_dim, hidden_dim) * np.sqrt(2 / (2 * hidden_dim)))
        self.bi = nn.Parameter(torch.rand(1, hidden_dim))
        # forget gate
        self.wxf = nn.Parameter(torch.rand(embedding_dim, hidden_dim) * np.sqrt(2 / (embedding_dim + hidden_dim)))
        self.whf = nn.Parameter(torch.rand(hidden_dim, hidden_dim) * np.sqrt(2 / (2 * hidden_dim)))
        self.bf = nn.Parameter(torch.rand(1, hidden_dim))
        # cell update
        self.wxc = nn.Parameter(torch.rand(embedding_dim, hidden_dim) * np.sqrt(2 / (embedding_dim + hidden_dim)))
        self.whc = nn.Parameter(torch.rand(hidden_dim, hidden_dim) * np.sqrt(2 / (2 * hidden_dim)))
        self.bc = nn.Parameter(torch.rand(1, hidden_dim))
        # output gate
        self.wxo = nn.Parameter(torch.rand(embedding_dim, hidden_dim) * np.sqrt(2 / (embedding_dim + hidden_dim)))
        self.who = nn.Parameter(torch.rand(hidden_dim, hidden_dim) * np.sqrt(2 / (2 * hidden_dim)))
        self.bo = nn.Parameter(torch.rand(1, hidden_dim))

        self.wy = nn.Parameter(torch.rand(hidden_dim, vocab_size) * np.sqrt(2 / (hidden_dim + vocab_size)))
        self.by = nn.Parameter(torch.rand(1, vocab_size))

    def forward(self, input_, hidden=None):
        sequence, batch = input_.size()
        if hidden is None:
            ht = input_.data.new(batch, self.hidden_dim).fill_(0).float()
            ct = input_.data.new(batch, self.hidden_dim).fill_(0).float()
        else:
            ht, ct = hidden
        embeds = self.embeddings(input_)
        rec = []
        for t in range(sequence):
            xt = embeds[t]
            it = torch.sigmoid(xt.mm(self.wxi) + ht.mm(self.whi) + self.bi)
            ft = torch.sigmoid(xt.mm(self.wxf) + ht.mm(self.whf) + self.bf)
            ct_ = torch.tanh(xt.mm(self.wxc) + ht.mm(self.whc) + self.bc)
            ot = torch.sigmoid(xt.mm(self.wxo) + ht.mm(self.who) + self.bo)
            ct = ft * ct + it * ct_
            ht = ot * torch.tanh(ct)
            Y = ht.mm(self.wy) + self.by
            rec.append(Y)
        return torch.cat(rec, 0), (ht, ct)

# 训练模型
def train():
    device = torch.device("cuda")
    vocab, full_data = data_process()
    vocab_size = len(vocab)
    full_data = torch.from_numpy(full_data)
    pick = int(full_data.size()[0] * 0.999)
    train_data, test_data = full_data[:pick,:], full_data[pick:,:]
    train_loader = DataLoader(train_data, batch_size=Config.batch_size, shuffle=True, drop_last=True)
    model = LSTM_model(vocab_size, Config.embedding_dim, Config.hidden_dim)
    # model.load_state_dict(torch.load('./poem.pth', 'cuda'))
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=Config.lr, momentum = 1.0)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(Config.epoch):
        cnt = 0
        for i, data in enumerate(train_loader):
            cnt += 1

            train_data = data.long().transpose(1, 0).contiguous().to(device)
            optimizer.zero_grad()
            # m, n = int(data.size()[1] * 0.8), data.size()[1]
            # train_data, test_data = data[:, :m], data[:, m:n]

            def get_loss(data):
                x, target = data[:-1, :], data[1:, :]
                output, _ = model(x)
                loss = criterion(output, target.contiguous().view(-1))
                return loss

            loss = get_loss(train_data)
            loss.backward()
            optimizer.step()

            test_loader = test_data.long().transpose(1, 0).contiguous().to(device)
            loss = get_loss(test_loader)
            perplexity = torch.mean(torch.exp(loss)).item()
            res_str = "[info] epoch: %d, Step: %d, Loss: %0.8f, Perplexity: %0.8f" % (epoch + 1, cnt, loss.item(), perplexity)
            print("[info] epoch: %d, Step: %d, Loss: %0.8f, Perplexity: %0.8f" % (epoch + 1, cnt, loss.item(), perplexity))
            with open('rec.txt', 'a', encoding='utf-8') as fp:
                fp.write(res_str + '\n')

    torch.save(model.state_dict(), './poem2.pth')


# 使用模型
def use():
    device = torch.device("cuda")
    vocab, full_data = data_process()
    vocab_size = len(vocab)
    model = LSTM_model(vocab_size, Config.embedding_dim, Config.hidden_dim)
    model.to(device)
    model.load_state_dict(torch.load('./poem.pth', 'cuda'))
    rec = model_use(model, '月', vocab)
    gen_poetry = ""
    for i in range(len(rec)):
        gen_poetry += rec[i]
    print(gen_poetry)

# 生成器
def model_use(model, begin, vocab):
    res = []
    input_ = (torch.Tensor([vocab.to_index(begin)]).view(1, 1).long()).cuda()
    hidden = None
    res.append(begin)

    cnt = 1
    for i in range(Config.max_gen_len - 1):
        output, hidden = model(input_, hidden)
        top_index = output.data[0].topk(10)
        pick = random.randint(0, 5)
        wordix = top_index[1][pick].item()
        def check(x):
            temp = vocab.to_word(x)
            if temp == ' ' or temp == '。' or temp == '？' or temp == '《' or temp == '》':
                return True
            if temp in res:
                return True
            return False
        while check(wordix):
            pick = random.randint(0, 5)
            wordix = top_index[1][pick].item()
        w = vocab.to_word(wordix)
        res.append(w)
        cnt += 1
        if cnt % 7 == 0:
            res.append('\n')
        input_ = input_.data.new([wordix]).view(1, 1)
    return res



# 调用train函数先进行训练，之后调用use函数进行生成
if __name__ == '__main__':
    train()
    # use()