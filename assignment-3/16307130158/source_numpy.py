from preprocess import build_data_vocab
from LSTM_torch import LSTM_torch as lstm_model
import torch.nn as nn
import torch
from torch.autograd import Variable
import json
import fastNLP
import numpy as np
from fastNLP import DataSet
from fastNLP import Instance
from fastNLP import Vocabulary
from fastNLP import Trainer
from fastNLP import Tester




def softmax(x):
    x = np.array(x)
    max_x = np.max(x)
    return np.exp(x-max_x) / np.sum(np.exp(x-max_x))

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def tanh(x):
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

class LSTM:
    def __init__(self, data_dim, hidden_dim):
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim

        self.whi, self.wxi, self.bi = self._init_wh_wx()
        self.whf, self.wxf, self.bf = self._init_wh_wx()
        self.who, self.wxo, self.bo = self._init_wh_wx()
        self.wha, self.wxa, self.ba = self._init_wh_wx()
        self.wy, self.by = np.random.uniform(-np.sqrt(1.0/self.hidden_dim), np.sqrt(1.0/self.hidden_dim),
                                             (self.data_dim, self.hidden_dim)), \
                           np.random.uniform(-np.sqrt(1.0/self.hidden_dim), np.sqrt(1.0/self.hidden_dim),
                                             (self.data_dim, 1))


    def _init_wh_wx(self):
        wh = np.random.uniform(-np.sqrt(1.0 / self.hidden_dim), np.sqrt(1.0 / self.hidden_dim),
                               (self.hidden_dim, self.hidden_dim))
        wx = np.random.uniform(-np.sqrt(1.0 / self.data_dim), np.sqrt(1.0 / self.data_dim),
                               (self.hidden_dim, self.data_dim))
        b = np.random.uniform(-np.sqrt(1.0 / self.data_dim), np.sqrt(1.0 / self.data_dim),
                              (self.hidden_dim, 1))
        return wh, wx, b

    def init_gate(self, T):
        iss = np.array([np.zeros((self.hidden_dim, 1))] * (T + 1))  # input gate
        fss = np.array([np.zeros((self.hidden_dim, 1))] * (T + 1))  # forget gate
        oss = np.array([np.zeros((self.hidden_dim, 1))] * (T + 1))  # output gate
        ass = np.array([np.zeros((self.hidden_dim, 1))] * (T + 1))  # current inputstate
        hss = np.array([np.zeros((self.hidden_dim, 1))] * (T + 1))  # hidden state
        css = np.array([np.zeros((self.hidden_dim, 1))] * (T + 1))  # cell state
        ys = np.array([np.zeros((self.data_dim, 1))] * T)  # output value
        return {'iss': iss, 'fss': fss, 'oss': oss,
                'ass': ass, 'hss': hss, 'css': css,
                'ys': ys}

    def cal_gate(self, wh, wx, b, ht_pre, x, activation_func):
        e = wh.dot(ht_pre)
        f = wx[:,x]
        res = e + f + b
        return activation_func(res)
        # return activation_func(wh.dot(ht_pre) + wx[:, x] + b)

    def forward(self, x):
        T = len(x)
        stats = self.init_gate(T)

        for t in range(T):
            ht_pre = np.array(stats['hss'][t-1]).reshape(-1, 1)

            # input gate
            stats['iss'][t] = self.cal_gate(self.whi, self.wxi, self.bi, ht_pre, x[t], sigmoid)
            # forget gate
            stats['fss'][t] = self.cal_gate(self.whf, self.wxf, self.bf, ht_pre, x[t], sigmoid)
            # output gate
            stats['oss'][t] = self.cal_gate(self.who, self.wxo, self.bo, ht_pre, x[t], sigmoid)
            # current inputstate
            stats['ass'][t] = self.cal_gate(self.wha, self.wxa, self.ba, ht_pre, x[t], tanh)

            # cell state, ct = ft * ct_pre + it * at
            stats['css'][t] = stats['fss'][t] * stats['css'][t - 1] + stats['iss'][t] * stats['ass'][t]
            # hidden state, ht = ot * tanh(ct)
            stats['hss'][t] = stats['oss'][t] * tanh(stats['css'][t])

            # output value, yt = softmax(self.wy.dot(ht) + self.by)
            stats['ys'][t] = softmax(self.wy.dot(stats['hss'][t]) + self.by)

        return stats

    def loss(self, x, y):
        cost = 0
        for i in range(len(y)):
            stats = self.forward(x[i])
            pre_yi = stats['ys'][range(len(y[i])), y[i]]
            cost -= np.sum(np.log(pre_yi))

        N = np.sum([len(yi) for yi in y])
        return cost/N

    def init_wh_wx_grad(self):
        dwh = np.zeros(self.whi.shape)
        dwx = np.zeros(self.wxi.shape)
        db = np.zeros(self.bi.shape)

        return dwh, dwx, db

    def cal_grad_delta(self, dwh, dwx, db, delta_net, ht_pre, x):
        dwh += delta_net * ht_pre
        dwx += delta_net * x
        db += delta_net

        return dwh, dwx, db

    def bptt(self, x, y):
        dwhi, dwxi, dbi = self.init_wh_wx_grad()
        dwhf, dwxf, dbf = self.init_wh_wx_grad()
        dwho, dwxo, dbo = self.init_wh_wx_grad()
        dwha, dwxa, dba = self.init_wh_wx_grad()
        dwy, dby = np.zeros(self.wy.shape), np.zeros(self.by.shape)

        delta_ct = np.zeros((self.hidden_dim, 1))

        stats = self.forward(x)
        delta_o = stats['ys']
        delta_o[np.arange(len(y)), y] -= 1

        for t in np.arange(len(y))[::-1]:

            dwy += delta_o[t].dot(stats['hss'][t].reshape(1, -1))
            dby += delta_o[t]

            # 目标函数对隐藏状态的偏导数
            delta_ht = self.wy.T.dot(delta_o[t])


            delta_ot = delta_ht * tanh(stats['css'][t])
            delta_ct += delta_ht * stats['oss'][t] * (1 - tanh(stats['css'][t]) ** 2)
            delta_it = delta_ct * stats['ass'][t]
            delta_ft = delta_ct * stats['css'][t - 1]
            delta_at = delta_ct * stats['iss'][t]

            delta_at_net = delta_at * (1 - stats['ass'][t] ** 2)
            delta_it_net = delta_it * stats['iss'][t] * (1 - stats['iss'][t])
            delta_ft_net = delta_ft * stats['fss'][t] * (1 - stats['fss'][t])
            delta_ot_net = delta_ot * stats['oss'][t] * (1 - stats['oss'][t])


            dwhf, dwxf, dbf = self.cal_grad_delta(dwhf, dwxf, dbf, delta_ft_net, stats['hss'][t - 1], x[t])
            dwhi, dwxi, dbi = self.cal_grad_delta(dwhi, dwxi, dbi, delta_it_net, stats['hss'][t - 1], x[t])
            dwha, dwxa, dba = self.cal_grad_delta(dwha, dwxa, dba, delta_at_net, stats['hss'][t - 1], x[t])
            dwho, dwxo, dbo = self.cal_grad_delta(dwho, dwxo, dbo, delta_ot_net, stats['hss'][t - 1], x[t])

        return dwhf, dwxf, dbf, dwhi, dwxi, dbi, dwha, dwxa, dba, dwho, dwxo, dbo, dwy, dby


    def sgd(self, x, y, learning_rate):
        dwhf, dwxf, dbf, dwhi, dwxi, dbi, dwha, dwxa, dba, dwho, dwxo, dbo, dwy, dby = self.bptt(x, y)
        self.whf, self.wxf, self.bf = self.update_wh_wx(learning_rate, self.whf, self.wxf, self.bf, dwhf, dwxf, dbf)
        self.whi, self.wxi, self.bi = self.update_wh_wx(learning_rate, self.whi, self.wxi, self.bi, dwhi, dwxi, dbi)
        self.wha, self.wxa, self.ba = self.update_wh_wx(learning_rate, self.wha, self.wxa, self.ba, dwha, dwxa, dba)
        self.who, self.wxo, self.bo = self.update_wh_wx(learning_rate, self.who, self.wxo, self.bo, dwho, dwxo, dbo)

        self.wy, self.by = self.wy - learning_rate * dwy, self.by - learning_rate * dby

    def update_wh_wx(self, learning_rate, wh, wx, b, dwh, dwx, db):
        wh -= learning_rate * dwh
        wx -= learning_rate * dwx
        b -= learning_rate * db
        return wh, wx, b

    def train(self, x_train, y_train, learning_rate = 0.005, n_epoch = 5):
        losses = []
        num_examples = 0

        for epoch in range(n_epoch):
            for i in range(len(y_train)):
                self.sgd(x_train[i], y_train[i], learning_rate)
                num_examples += 1

            loss = self.loss(x_train, y_train)
            losses.append(loss)
            print('epoch {0}: loss = {1}'.format(epoch + 1, loss))
            if len(losses) > 1 and losses[-1] > losses[-2]:
                learning_rate *= 0.5
                print('decrease learning_rate to ', learning_rate)

