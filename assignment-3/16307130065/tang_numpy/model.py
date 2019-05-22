import numpy as np
import torch.nn as nn

def softmax(x):
    x = np.array(x)
    max_x = np.max(x)
    return np.exp(x-max_x) / np.sum(np.exp(x-max_x))


def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))


def tanh(x):
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))


class Poetry:
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.W_hi, self.W_xi, self.b_i = self._init_param()
        self.W_hf, self.W_xf, self.b_f = self._init_param()
        self.W_ho, self.W_xo, self.b_o = self._init_param()
        self.W_ha, self.W_xa, self.b_a = self._init_param()
        self.w = np.random.uniform(-np.sqrt(1.0/self.hidden_dim), np.sqrt(1.0/self.hidden_dim), (self.vocab_size, self.hidden_dim))
        self.b = np.random.uniform(-np.sqrt(1.0/self.hidden_dim), np.sqrt(1.0/self.hidden_dim), (self.vocab_size, 1))

    def _init_param(self):
        W_h = np.random.uniform(-np.sqrt(1.0/self.hidden_dim), np.sqrt(1.0/self.hidden_dim), (self.hidden_dim, self.hidden_dim))
        W_x = np.random.uniform(-np.sqrt(1.0/self.vocab_size), np.sqrt(1.0/self.vocab_size), (self.hidden_dim, self.vocab_size))
        b = np.random.uniform(-np.sqrt(1.0/self.vocab_size), np.sqrt(1.0/self.vocab_size), (self.hidden_dim, 1))

        return W_h, W_x, b

    def _init_state(self, seq_len):
        i_state = np.array([np.zeros((self.hidden_dim, 1))]*(seq_len+1))
        f_state = np.array([np.zeros((self.hidden_dim, 1))]*(seq_len+1))
        o_state = np.array([np.zeros((self.hidden_dim, 1))]*(seq_len+1))
        a_state = np.array([np.zeros((self.hidden_dim, 1))]*(seq_len+1))
        h_state = np.array([np.zeros((self.hidden_dim, 1))]*(seq_len+1))
        c_state = np.array([np.zeros((self.hidden_dim, 1))]*(seq_len+1))
        output = np.array([np.zeros((self.vocab_size, 1))]*seq_len)

        return {'i': i_state, 'f': f_state, 'o': o_state,'a': a_state, 'h': h_state, 'c': c_state,'out': output}

    def forward(self, x):
        seq_len = len(x)
        stats = self._init_state(seq_len)

        for t in range(seq_len):
            h_t = np.array(stats['h'][t-1]).reshape(-1, 1)

            stats['i'][t] = self.gate(self.W_hi, self.W_xi, self.b_i, h_t, x[t], sigmoid)
            stats['f'][t] = self.gate(self.W_hf, self.W_xf, self.b_f, h_t, x[t], sigmoid)
            stats['o'][t] = self.gate(self.W_ho, self.W_xo, self.b_o, h_t, x[t], sigmoid)
            stats['a'][t] = self.gate(self.W_ha, self.W_xa, self.b_a, h_t, x[t], tanh)
            stats['c'][t] = stats['f'][t] * stats['c'][t-1] + stats['i'][t] * stats['a'][t]
            stats['h'][t] = stats['o'][t] * tanh(stats['c'][t])
            stats['out'][t] = softmax(self.w.dot(stats['h'][t]) + self.b)

        return stats

    def gate(self, W_h, W_x, b, h_t_pre, x, func):
        return func(W_h.dot(h_t_pre) + W_x[:, x].reshape(-1,1) + b)

    def predict(self, x):
        stats = self.forward(x)
        pre = np.argmax(stats['out'].reshape(len(x), -1), axis=1)
        return pre

    def loss(self, x, y):
        cost = 0
        for i in range(len(y)):
            stats = self.forward(x[i])
            pre_i = stats['out'][range(len(y[i])), y[i]]
            cost -= np.sum(np.log(pre_i))

        sum = np.sum([len(yi) for yi in y])
        ave_loss = cost / sum

        return ave_loss

    def _init_grad(self):
        dW_h = np.zeros(self.W_hi.shape)
        dW_x = np.zeros(self.W_xi.shape)
        db = np.zeros(self.b_i.shape)

        return dW_h, dW_x, db

    def BPTT(self, x, y):
        dW_hi, dW_xi, db_i = self._init_grad()
        dW_hf, dW_xf, db_f = self._init_grad()
        dW_ho, dW_xo, db_o = self._init_grad()
        dW_ha, dW_xa, db_a = self._init_grad()
        dw = np.zeros(self.w.shape)
        db = np.zeros(self.b.shape)

        delta_ct = np.zeros((self.hidden_dim, 1))

        stats = self.forward(x)
        delta_o = stats['output']
        delta_o[np.arange(len(y)), y] -= 1

        for t in np.arange(len(y))[::-1]:
            dw += delta_o[t].dot(stats['h'][t].reshape(1, -1))
            db += delta_o[t]

            delta_ht = self.w.T.dot(delta_o[t])

            delta_ot = delta_ht * tanh(stats['c'][t])
            delta_ct += delta_ht * stats['o'][t] * (1-tanh(stats['c'][t])**2)
            delta_it = delta_ct * stats['a'][t]
            delta_ft = delta_ct * stats['c'][t-1]
            delta_at = delta_ct * stats['i'][t]

            delta_a = delta_at * (1-stats['a'][t]**2)
            delta_i = delta_it * stats['i'][t] * (1-stats['i'][t])
            delta_f = delta_ft * stats['f'][t] * (1-stats['f'][t])
            delta_o = delta_ot * stats['o'][t] * (1-stats['o'][t])

            dW_hf, dW_xf, db_f = self._cal_grad(dW_hf, dW_xf, db_f, delta_f, stats['h'][t-1], x[t])
            dW_hi, dW_xi, db_i = self._cal_grad(dW_hi, dW_xi, db_i, delta_i, stats['h'][t-1], x[t])
            dW_ha, dW_xa, db_a = self._cal_grad(dW_ha, dW_xa, db_a, delta_a, stats['h'][t-1], x[t])
            dW_ho, dW_xo, db_o = self._cal_grad(dW_ho, dW_xo, db_o, delta_o, stats['h'][t-1], x[t])

        return [dW_hf, dW_xf, db_f, dW_hi, dW_xi, db_i, dW_ha, dW_xa, db_a, dW_ho, dW_xo, db_o, dw, db]

    def _cal_grad(self, dW_h, dW_x, db, delta, h_t, x):
        dW_h += delta * h_t
        dW_x += delta * x
        db += delta

        return dW_h, dW_x, db

    def cal_grad(self, x, y, lr):
        dW_hf, dW_xf, db_f, dW_hi, dW_xi, db_i, dW_ha, dW_xa, db_a, dW_ho, dW_xo, db_o, dw, db = self.BPTT(x, y)

        self.W_hf, self.W_xf, self.b_f = self._update_param(lr, self.W_hf, self.W_xf, self.b_f, dW_hf, dW_xf, db_f)
        self.W_hi, self.W_xi, self.b_i = self._update_param(lr, self.W_hi, self.W_xi, self.b_i, dW_hi, dW_xi, db_i)
        self.W_ha, self.W_xa, self.b_a = self._update_param(lr, self.W_ha, self.W_xa, self.b_a, dW_ha, dW_xa, db_a)
        self.W_ho, self.W_xo, self.b_o = self._update_param(lr, self.W_ho, self.W_xo, self.b_o, dW_ho, dW_xo, db_o)
        self.w = self.w - lr * dw
        self.b = self.b - lr * db

    def _update_param(self, learning_rate, W_h, W_x, b, dW_h, dW_x, db):
        W_h -= learning_rate * dW_h
        W_x -= learning_rate * dW_x
        b -= learning_rate * db

        return W_h, W_x, b

    def train(self, inp, out, lr=0.005, epoch=5):
        LossList = []
        embeds = self.embeddings(inp)
        for ep in range(epoch):
            for i in range(len(out)):
                self.cal_grad(embeds[i], out[i], lr)

            loss = self.loss(inp, out)
            LossList.append(loss)
            print('epoch %d loss %0.8f' % (epoch+1, loss))
            if len(LossList) > 1 and LossList[-1] > LossList[-2]:
                lr *= 0.5
