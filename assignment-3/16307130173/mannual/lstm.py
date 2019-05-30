import numpy as np
import word2vec
import torch
from torch import nn
from torch.autograd import Variable

def softmax(x):
    x = np.array(x)
    return np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

class LSTM:
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, load_path = None):

        self.embed = torch.nn.Embedding(input_dim, embedding_dim)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.Wi_h, self.Wi_x, self.bi = self.init_weight()
        self.Wf_h, self.Wf_x, self.bf = self.init_weight()
        self.Wo_h, self.Wo_x, self.bo = self.init_weight()
        self.Wc_h, self.Wc_x, self.bc = self.init_weight()  # for current input state
        self.Wy = np.random.uniform(-np.sqrt(1 / self.hidden_dim), np.sqrt(1 / self.hidden_dim),
                                    (self.output_dim, self.hidden_dim))
        self.by = np.random.uniform(-np.sqrt(1 / self.hidden_dim), np.sqrt(1 / self.hidden_dim),
                                    (self.output_dim, 1))

        if load_path != None:
            self.load(load_path)

    def save(self, file_path):
        np.savez(file_path, Wi_h = self.Wi_h, Wi_x = self.Wi_x, bi = self.bi,
                 Wf_h = self.Wf_h, Wf_x = self.Wf_x, bf = self.bf,
                 Wo_h = self.Wo_h, Wo_x = self.Wo_x, bo = self.bo,
                 Wc_h = self.Wc_h, Wc_x = self.Wc_x, bc = self.bc,
                 Wy = self.Wy, by = self.by)

    def load(self, file_path):
        r = np.load(file_path)
        self.Wi_h, self.Wi_x, self.bi = r['Wi_h'], r['Wi_x'], r['bi']
        self.Wf_h, self.Wf_x, self.bf = r['Wf_h'], r['Wf_x'], r['bf']
        self.Wo_h, self.Wo_x, self.bo = r['Wo_h'], r['Wo_x'], r['bo']
        self.Wc_h, self.Wc_x, self.bc = r['Wc_h'], r['Wc_x'], r['bc']
        self.Wy, self.by = r['Wy'], r['by']

    def init_weight(self):
        tmph = np.sqrt(1.0 / self.hidden_dim)
        tmpe = np.sqrt(1.0 / self.embedding_dim)

        h = np.random.uniform(-tmph, tmph, (self.hidden_dim, self.hidden_dim))
        x = np.random.uniform(-tmpe, tmpe, (self.hidden_dim, self.embedding_dim))
        b = np.random.uniform(-tmpe, tmpe, (self.hidden_dim, 1))

        return h, x, b

    def init_status(self, T):
        ig = np.array([np.zeros((self.hidden_dim, 1))] * (T + 1))
        fg = np.array([np.zeros((self.hidden_dim, 1))] * (T + 1))
        og = np.array([np.zeros((self.hidden_dim, 1))] * (T + 1))
        cig = np.array([np.zeros((self.hidden_dim, 1))] * (T + 1))
        hg = np.array([np.zeros((self.hidden_dim, 1))] * (T + 1))
        cg = np.array([np.zeros((self.hidden_dim, 1))] * (T + 1))
        yg = np.array([np.zeros((self.output_dim, 1))] * T)

        return {'ig': ig, 'fg': fg, 'og': og, 'cig': cig, 'hg': hg, 'cg': cg, 'yg': yg}

    def foward(self, x):
        '''
            前向传播：计算各个门的状态
        '''
        T = x.shape[0]
        status = self.init_status(T)

        for t in range(T):
            h_pre = np.array(status['hg'][t - 1].reshape(-1, 1))
            xt = np.array(x[t].reshape(-1, 1))

            status['ig'][t] = self.gate(self.Wi_h, self.Wi_x, self.bi, h_pre, xt, sigmoid)
            status['fg'][t] = self.gate(self.Wf_h, self.Wf_x, self.bf, h_pre, xt, sigmoid)
            status['og'][t] = self.gate(self.Wo_h, self.Wo_x, self.bo, h_pre, xt, sigmoid)
            status['cig'][t] = self.gate(self.Wc_h, self.Wc_x, self.bc, h_pre, xt, sigmoid)
            status['cg'][t] = status['fg'][t] * status['cg'][t - 1] + status['ig'][t] * status['cig'][t]
            status['hg'][t] = status['og'][t] * tanh(status['cg'][t])
            status['yg'][t] = softmax(self.Wy.dot(status['hg'][t]) + self.by)

        return status

    def gate(self, wh, wx, b, h_pre, x, activate):
        return activate(wh.dot(h_pre) + wx.dot(x) + b)

    def predict(self, x):
        status = self.foward(x)
        pref = np.argmax(status['yg'].reshape(self.output_dim, -1), axis=1)
        return pref

    def loss(self, X, y):
        '''
            目标函数：softmax交叉熵损失函数
        '''
        cost = 0
        length_y = len(y)
        for i in range(length_y):
            status = self.foward(X[i])
            cost -= np.sum(np.log(status['yg'][range(len(y[i])), y[i]]))

        return cost / np.sum([len(yi) for yi in y])

    def init_grad(self):
        dwh = np.zeros(self.Wi_h.shape)
        dwx = np.zeros(self.Wi_x.shape)
        db = np.zeros(self.bi.shape)

        return dwh, dwx, db

    def get_grad(self, x, y):
        '''
            计算梯度
        '''
        dwi_h, dwi_x, dbi = self.init_grad()
        dwf_h, dwf_x, dbf = self.init_grad()
        dwc_h, dwc_x, dbc = self.init_grad()
        dwo_h, dwo_x, dbo = self.init_grad()
        dwy, dby = np.zeros(self.Wy.shape), np.zeros(self.by.shape)

        delta_ct = np.zeros((self.hidden_dim, 1))
        status = self.foward(x)

        len_y = len(y)
        delta_y = status['yg']
        delta_y[np.arange(len_y), y] -= 1


        for t in np.arange(len_y)[::-1]:
            dwy += delta_y[t].dot(status['hg'][t].reshape(1, -1))
            dby += delta_y[t]
            xt = np.array(x[t]).reshape(-1, 1)

            delta_ht = self.Wy.T.dot(delta_y[t])
            delta_ot = delta_ht * tanh(status['cg'][t]) * status['og'][t] * (1 - status['og'][t])

            delta_ct += delta_ht * status['og'][t] * (1 - tanh(status['cg'][t]) ** 2)
            delta_cit = delta_ct * status['ig'][t] * (1 - status['cig'][t] ** 2)
            delta_it = delta_ct * status['cig'][t] * status['ig'][t] * (1 - status['ig'][t])
            delta_ft = delta_ct * status['cg'][t - 1] * status['fg'][t] * (1 - status['fg'][t])

            dwi_h, dwi_x, dbi = self.grad_delta(dwi_h, dwi_x, dbi, delta_it, status['hg'][t - 1], xt)
            dwf_h, dwf_x, dbf = self.grad_delta(dwf_h, dwf_x, dbf, delta_ft, status['hg'][t - 1], xt)
            dwc_h, dwc_x, dbc = self.grad_delta(dwc_h, dwc_x, dbc, delta_cit, status['hg'][t - 1], xt)
            dwo_h, dwo_x, dbo = self.grad_delta(dwo_h, dwo_x, dbo, delta_ot, status['hg'][t - 1], xt)

        return [dwf_h, dwf_x, dbf, dwi_h, dwi_x, dbi,
                dwc_h, dwc_x, dbc, dwo_h, dwo_x, dbo, dwy, dby]

    def grad_delta(self, dh, dx, db, delta, ht, x):
        dh += delta * ht
        dx += delta * x
        db += delta

        return dh, dx, db

    def cal_grad(self, x, y, learning_rate):
        '''
            更新梯度
        '''
        dwf_h, dwf_x, dbf, dwi_h, dwi_x, dbi, \
        dwc_h, dwc_x, dbc, dwo_h, dwo_x, dbo, dwy, dby = self.get_grad(x, y)

        self.Wi_h, self.Wi_x, self.bi = self.update_weight(learning_rate, self.Wi_h, self.Wi_x, self.bi, dwi_h, dwi_x, dbi)
        self.Wf_h, self.Wf_x, self.bf = self.update_weight(learning_rate, self.Wf_h, self.Wf_x, self.bf, dwf_h, dwf_x, dbf)
        self.Wc_h, self.Wc_x, self.bc = self.update_weight(learning_rate, self.Wc_h, self.Wc_x, self.bc, dwc_h, dwc_x, dbc)
        self.Wo_h, self.Wo_x, self.bo = self.update_weight(learning_rate, self.Wo_h, self.Wo_x, self.bo, dwo_h, dwo_x, dbo)

    def update_weight(self, learning_rate, wh, wx, b, dh, dx, db):
        wh -= learning_rate * dh
        wx -= learning_rate * dx
        b -= learning_rate * db

        return wh, wx, b

    def embedding(self, X):
        length = X.shape[0]
        xi = []
        for i in range(length):
            lst = []
            for j in X[i]:
                tmp = self.embed(Variable(torch.LongTensor([j])))
                lst.append(tmp.detach().numpy())
            xi.append(lst)
        xi = np.array(xi).reshape((X.shape[0], X.shape[1], self.embedding_dim))

        return xi

    def train(self, X, y, test_X, test_y, learning_rate = 0.005, num = 5):
        losses = []
        length_y = len(y)
        print(X.shape)
        print(self.embed)

        xi = self.embedding(X)
        test_xi = self.embedding(test_X)

        for nn in range(num):
            for i in range(length_y):
                self.cal_grad(xi[i], y[i], learning_rate)

            loss = self.loss(test_xi, test_y)
            losses.append(loss)

            print('No. %s: loss = %s' % (str(nn), str(loss)))


word2vec = word2vec.word2Vec('./data.txt')
ds_X, ds_y = word2vec.get_dataset()

n = ds_X.shape[0]

train_X, train_y = ds_X[:int(n * 0.8)], ds_y[:int(n * 0.8)]
test_X, test_y = ds_X[int(n * 0.8): n], ds_y[int(n * 0.8): n]

lstm = LSTM(word2vec.dict_size, 120, 120, word2vec.dict_size)

for i in range(n)[::200]:
    lstm.train(ds_X[i:i + 200], ds_y[i:i + 200], test_X, test_y, learning_rate=0.01, num = 100)
    lstm.save('./model.npz')
