import numpy as np
from fastNLP import Vocabulary, DataSet, Instance
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import array

def get_dataset(file):
    dataset = DataSet()
    with open(file, "r") as f:
        data = ""
        for line in f:
            if line == '\n':
                dataset.append(Instance(poem=data))
                data = ""
            else:
                data = data + line.strip()  # 去掉\n
    train_data, dev_data = dataset.split(0.2)
    return train_data, dev_data


def get_vocabulary(train_data, test_data):
    # 构建词表, Vocabulary.add(word)
    vocab = Vocabulary(min_freq=2, unknown='<unk>', padding='<pad>')
    train_data.apply(lambda x: [vocab.add(word) for word in x['poem']])
    vocab.build_vocab()
    # index句子, Vocabulary.to_index(word)
    train_data.apply(lambda x: [vocab.to_index(word) for word in x['poem']], new_field_name='words')
    test_data.apply(lambda x: [vocab.to_index(word) for word in x['poem']], new_field_name='words')
    print(train_data[0])
    print(vocab.word2idx)
    return vocab, train_data, test_data


def softmax(x):
    x = np.exp(x)
    return x/sum(x)


def sigmod(x):
    return 1/(1+np.exp(-x))


def tanh(x):
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))


def de_sigmod(x):
    return x*(1-x)


def de_tanh(x):
    return 1 - x*x

def gate(Wh, Wx, h, x, b, active=sigmod):
    net = np.dot(Wh, h) + np.dot(Wx, x) + b
    return active(net)


class LSM:
    def __init__(self):
        self.time = 0
        self.vec_c = []
        self.vec_h = []
        self.vec_f = []
        self.vec_i = []
        self.vec_o = []
        self.vec_c2 = []
        self.h_delta = []
        self.c_delta = []
        self.f_delta = []
        self.i_delta = []
        self.o_delta = []
        self.c2_delta = []
        # 遗忘门权重矩阵Wfh, Wfx, 偏置项bf
        self.Wfh, self.Wfx, self.bf = (
            self.init_weight_mat())
        # 输入门权重矩阵Wfh, Wfx, 偏置项bf
        self.Wih, self.Wix, self.bi = (
            self.init_weight_mat())
        # 输出门权重矩阵Wfh, Wfx, 偏置项bf
        self.Woh, self.Wox, self.bo = (
            self.init_weight_mat())
        # 单元状态权重矩阵Wfh, Wfx, 偏置项bf
        self.Wch, self.Wcx, self.bc = (
            self.init_weight_mat())

    def init_weight_mat(self):
        '''
        初始化权重矩阵
        '''
        Wh = np.random.uniform(-1e-4, 1e-4,
                               (self.state_width, self.state_width))
        Wx = np.random.uniform(-1e-4, 1e-4,
                               (self.state_width, self.input_width))
        b = np.zeros((self.state_width, 1))
        return Wh, Wx, b

    def forward(self, x):
        self.time += 1
        h_before = self.vec_h[self.time - 1] #等会儿补充
        # forget gate
        Ft = gate(self.Wfh, self.Wfx, h_before, x, self.bf)
        self.vec_f.append(Ft)
        # input gate
        It = gate(self.Wih, self.Wix, h_before, x, self.bi)
        self.vec_i.append()
        # 当前输入单元状态
        Ct = gate(self.Wch, self.Wcx, h_before, x, self.bc, active=tanh)
        self.vec_c2.append(Ct)
        # 当前时刻的单元状态
        c_before = self.vec_c[self.time - 1]
        c = Ft * c_before + It * Ct
        self.vec_c.append(c)
        self.vec_c.append(c)
        # 输出门
        Ot = gate(self.Woh, self.Wox, h_before, x, self.bo)
        self.vec_o.append(Ot)
        # 最终输出
        h = Ot * tanh(c)
        self.vec_h.append(h)

    def delta(self, t):
        h_delta = self.h_delta[t]
        c = self.vec_c[t]
        o = self.vec_o[t]
        f = self.vec_f[t]
        i = self.vec_i[t]
        c2 = self.vec_c2[t]
        c_before = self.vec_c[t-1]
        o_delta = h_delta * tanh(c)*de_sigmod(o)
        f_delta = h_delta*o*de_tanh(tanh(c))*c_before*de_sigmod(f)
        i_delta = h_delta*o*de_tanh(tanh(c))*c2*de_sigmod(i)
        c2_delta = h_delta*o*de_tanh(tanh(c))*i*de_tanh(c2)
        h_delta_before = (
                np.dot(o_delta, self.Woh) +
                np.dot(i_delta, self.Wih) +
                np.dot(f_delta, self.Wfh) +
                np.dot(c2_delta, self.Wch)
        ).transpose()
        self.c2_delta[t] = c2_delta
        self.o_delta[t] = o_delta
        self.f_delta[t] = f_delta
        self.i_delta[t] = i_delta
        self.h_delta[t-1] = h_delta_before

    def grad(self, t):
        h_before = self.vec_h[t-1]
        oh_grad = np.dot(self.o_delta[t], h_before.T)
        fh_grad = np.dot(self.f_delta[t], h_before.T)
        ih_grad = np.dot(self.i_delta[t], h_before.T)
        c2h_grad = np.dot(self.c2_delta[t], h_before.T)
        bo_grad = self.o_delta[t]
        bi_grad = self.i_delta[t]
        bf_grad = self.f_delta[t]
        bc2_grad = self.c2_delta[t]
        return oh_grad, fh_grad, ih_grad, c2h_grad,bo_grad, bi_grad, bf_grad,bc2_grad


    def gradient(self,x):
        for t in range(self.times, 0, -1):
            # 计算各个时刻的梯度
            (Woh_grad, Wfh_grad, Wih_grad, Wc2h_grad,
             bo_grad, bi_grad, bf_grad, bc2_grad) = (
                self.grad(t))
            # 实际梯度是各时刻梯度之和
            self.Wfh_grad += Wfh_grad
            self.bf_grad += bf_grad
            self.Wih_grad += Wih_grad
            self.bi_grad += bi_grad
            self.Woh_grad += Woh_grad
            self.bo_grad += bo_grad
            self.Wch_grad += Wc2h_grad
            self.bc_grad += bc2_grad
            # 计算对本次输入x的权重梯度
        xt = x.transpose()
        self.Wfx_grad = np.dot(self.f_delta[-1], xt)
        self.Wix_grad = np.dot(self.i_delta[-1], xt)
        self.Wox_grad = np.dot(self.o_delta[-1], xt)
        self.Wcx_grad = np.dot(self.c2_delta[-1], xt)

    def update(self):
        '''
        按照梯度下降，更新权重
        '''
        self.Wfh -= self.learning_rate * self.Whf_grad
        self.Wfx -= self.learning_rate * self.Whx_grad
        self.bf -= self.learning_rate * self.bf_grad
        self.Wih -= self.learning_rate * self.Whi_grad
        self.Wix -= self.learning_rate * self.Whi_grad
        self.bi -= self.learning_rate * self.bi_grad
        self.Woh -= self.learning_rate * self.Wof_grad
        self.Wox -= self.learning_rate * self.Wox_grad
        self.bo -= self.learning_rate * self.bo_grad
        self.Wch -= self.learning_rate * self.Wcf_grad
        self.Wcx -= self.learning_rate * self.Wcx_grad
        self.bc -= self.learning_rate * self.bc_grad





def main():
    #train_data, dev_data = get_dataset("tangshi.txt")
    #vocabulary, train_data, dev_data = get_vocabulary(train_data, dev_data)
    #dict_size = len(vocabulary)
    #embeds = nn.Embedding(dict_size, 300)
    #vector = {}
    #for key in vocabulary.word2idx:
     #   idx = torch.LongTensor([vocabulary.word2idx[key]])
      #  vector[int(idx)] = embeds(Variable(idx)).detach().numpy()[0]
    print(softmax([2,3,4]))
    #print(forget_gate([1,2,3],[1,2,3],[1,2,3],[1,2,3]))


if __name__ == "__main__":
    main()