import numpy as np
from config import Config


config = Config()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dsigmoid(x):
    return x * (1 - x)


def tanh(x):
    return np.tanh(x)


def dtanh(x):
    return 1 - x * x


class Parameter(object):
    def __init__(self, name, value):
        self.name = name
        self.v = value
        self.d = np.zeros_like(value)
        self.m = np.zeros_like(value)
        self.g = np.zeros_like(value)
        self.t = 0


class LSTMCell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        stdv = 1.0 / np.sqrt(self.hidden_size)
        z_size = self.hidden_size + self.input_size
        self.W_f = Parameter('W_f', np.random.uniform(-stdv, stdv, size=(self.hidden_size, z_size)))
        self.b_f = Parameter('b_f', np.zeros((self.hidden_size, 1)))
        self.W_i = Parameter('W_i', np.random.uniform(-stdv, stdv, size=(self.hidden_size, z_size)))
        self.b_i = Parameter('b_i', np.zeros((self.hidden_size, 1)))
        self.W_C = Parameter('W_C', np.random.uniform(-stdv, stdv, size=(self.hidden_size, z_size)))
        self.b_C = Parameter('b_C', np.zeros((self.hidden_size, 1)))
        self.W_o = Parameter('W_o', np.random.uniform(-stdv, stdv, size=(self.hidden_size, z_size)))
        self.b_o = Parameter('b_o', np.zeros((self.hidden_size, 1)))
        self.W_v = Parameter('W_v', np.random.uniform(-stdv, stdv, size=(self.input_size, self.hidden_size)))
        self.b_v = Parameter('b_v', np.zeros((self.input_size, 1)))

    def all(self):
        return [self.W_f, self.W_i, self.W_C, self.W_o, self.W_v, self.b_f, self.b_i, self.b_C, self.b_o, self.b_v]

    def clear_gradients(self):
        for i in self.all():
            i.d.fill(0)

    def forward(self, x, h_prev, C_prev):
        z = np.row_stack((h_prev, x))
        f = sigmoid(np.dot(self.W_f.v, z) + self.b_f.v)
        i = sigmoid(np.dot(self.W_i.v, z) + self.b_i.v)
        C_bar = tanh(np.dot(self.W_C.v, z) + self.b_C.v)
        C = f * C_prev + i * C_bar
        o = sigmoid(np.dot(self.W_o.v, z) + self.b_o.v)
        h = o * tanh(C)
        v = np.dot(self.W_v.v, h) + self.b_v.v
        y = np.exp(v) / np.sum(np.exp(v))

        return z, f, i, C_bar, C, o, h, v, y

    def backward(self, target, dh_next, dC_next, C_prev, z, f, i, C_bar, C, o, h, v, y, ):
        dv = np.copy(y)
        dv[target] -= 1

        self.W_v.d += np.dot(dv, h.T)
        self.b_v.d += dv

        dh = np.dot(self.W_v.v.T, dv)
        dh += dh_next
        do = dh * tanh(C)
        do = dsigmoid(o) * do
        self.W_o.d += np.dot(do, z.T)
        self.b_o.d += do

        dC = np.copy(dC_next)
        dC += dh * o * dtanh(tanh(C))
        dC_bar = dC * i
        dC_bar = dtanh(C_bar) * dC_bar
        self.W_C.d += np.dot(dC_bar, z.T)
        self.b_C.d += dC_bar

        di = dC * C_bar
        di = dsigmoid(i) * di
        self.W_i.d += np.dot(di, z.T)
        self.b_i.d += di

        df = dC * C_prev
        df = dsigmoid(f) * df
        self.W_f.d += np.dot(df, z.T)
        self.b_f.d += df

        dz = (np.dot(self.W_f.v.T, df) + np.dot(self.W_i.v.T, di) + np.dot(self.W_C.v.T, dC_bar) + np.dot(self.W_o.v.T, do))
        dh_prev = dz[:self.hidden_size, :]
        dC_prev = f * dC
        return dh_prev, dC_prev


class LSTMNumpy:
    def __init__(self, input_size, hidden_size):
        self.lstm_cell = LSTMCell(input_size, hidden_size)

    def forward_backward(self, inputs, targets, h_prev, C_prev, a=0.9, b=1):
        x_s, z_s, f_s, i_s, C_bar_s, C_s, o_s, h_s, v_s, y_s = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}

        h_s[-1] = np.copy(h_prev)
        C_s[-1] = np.copy(C_prev)

        loss = 0
        seq_len = inputs.shape[0]
        for t in range(seq_len):
            x_s[t] = np.array(inputs[t]).T
            (z_s[t], f_s[t], i_s[t], C_bar_s[t], C_s[t], o_s[t], h_s[t], v_s[t], y_s[t]) = self.lstm_cell.forward(x_s[t], h_s[t - 1], C_s[t - 1])
            loss += -np.log(y_s[t][targets[t], 0])

        self.lstm_cell.clear_gradients()

        dh_next = np.zeros_like(h_s[0])
        dC_next = np.zeros_like(C_s[0])

        for t in reversed(range(seq_len)):
            dh_next, dC_next = self.lstm_cell.backward(target=targets[t], dh_next=dh_next, dC_next=dC_next, C_prev=C_s[t - 1], z=z_s[t], f=f_s[t], i=i_s[t], C_bar=C_bar_s[t], C=C_s[t], o=o_s[t], h=h_s[t], v=v_s[t], y=y_s[t])

        # Adam
        for i in self.lstm_cell.all():
            i.t += 1
            i.m = a * i.m + (1 - a) * i.d
            i.g = b * i.g + (1 - b) * i.d * i.d
            i.v -= config.lr * i.m / (1 - a ** i.t) / (np.sqrt(i.g / (1 - b ** i.t) + 1e-8))

        return loss, h_s[seq_len - 1], C_s[seq_len - 1]
