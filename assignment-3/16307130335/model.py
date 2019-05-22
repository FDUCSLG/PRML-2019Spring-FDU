import numpy as np
from torch import nn, tensor
from torch.autograd import Variable

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))


def tanh(x):
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))


def softmax(x):
    x = np.array(x)
    max_x = np.max(x)
    return np.exp(x-max_x) / np.sum(np.exp(x-max_x))


class LSTM:

    def __init__(self, input_dim, hidden_dim, vocab_dim):
        # input_dim: 词向量维度； hidden_dim:隐藏层以及状态层维度；vocab_dim:词典维度
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.vocab_dim = vocab_dim
        self.embedding = nn.Embedding(vocab_dim, input_dim)
        # 初始化参数矩阵
        self.whi, self.wxi, self.bi = self._init_wh_wx()
        self.whf, self.wxf, self.bf = self._init_wh_wx()
        self.who, self.wxo, self.bo = self._init_wh_wx()
        self.whc, self.wxc, self.bc = self._init_wh_wx()
        self.wy = self._init_w((vocab_dim, hidden_dim), hidden_dim)
        self.by = self._init_w((vocab_dim, 1), hidden_dim)

    def _init_w(self, shape, input_dim):
        w = np.random.uniform(-np.sqrt(1.0/input_dim), np.sqrt(1.0/input_dim), shape)
        return w

    def _init_wh_wx(self):
        wh = self._init_w((self.hidden_dim, self.hidden_dim), self.hidden_dim)
        wx = self._init_w((self.hidden_dim, self.input_dim), self.input_dim)
        b = self._init_w((self.hidden_dim, 1), self.input_dim)
        return wh, wx, b

    def _init_list(self, shape, length):

        return np.array([np.zeros(shape)]*length)

    def _init_state(self, T, hist={'h': 0, 'c': 0}):
        gate_shape = (self.hidden_dim, 1)
        output_shape = (self.vocab_dim, 1)
        i = self._init_list(gate_shape, T + 1)  # 输入门
        f = self._init_list(gate_shape, T + 1)  # 遗忘门
        o = self._init_list(gate_shape, T + 1)  # 输出门
        h = self._init_list(gate_shape, T + 1)  # 隐藏层输出
        c = self._init_list(gate_shape, T + 1)  # 单元状态
        c_ = self._init_list(gate_shape, T + 1)  # 当前输入状态 ～c
        y = self._init_list(output_shape, T)  # 最终输出
        # 将上一次的状态存放在（T+1)时刻
        h[-1] = hist['h']
        c[-1] = hist['c']
        return {'i': i, 'f': f, 'o': o, 'h': h, 'c': c, 'c_': c_, 'y': y}

    def _init_delta_gate(self):
        dwh = np.zeros((self.hidden_dim, self.hidden_dim))
        dwx = np.zeros((self.hidden_dim, self.input_dim))
        db = np.zeros((self.hidden_dim, 1))
        return dwh, dwx, db

    def new_gate(self, Wh, Wx, b, h_pre, x, activation):
        tensor_x = self.embedding(Variable(tensor(x)))
        x = tensor_x.tolist()
        x = np.array(x).reshape(-1, 1)/1000
        return activation(Wh.dot(h_pre) + Wx.dot(x.reshape(-1, 1)) + b)

    def delta_grad(self, dwh, dwx, db, delta, h_pre, x):
        dwh += delta * h_pre
        dwx += delta * x
        db += delta
        return dwh, dwx, db

    def update_wh_wx(self, learning_rate, wh, wx, b, dwh, dwx, db):
        wh -= learning_rate * dwh
        wx -= learning_rate * dwx
        b -= learning_rate * db
        return wh, wx, b

    def forward(self, x, hist={'h': 0, 'c': 0}):
        # 向量长度
        T = len(x)
        # 初始化各个状态向量
        state = self._init_state(T, hist)
        for t in range(T):
            # h(t-1), reshape(-1, 1)代表将其转换为列向量
            h_pre = np.array(state['h'][t - 1]).reshape(-1, 1)
            # 输入门
            state['i'][t] = self.new_gate(self.whi, self.wxi, self.bi, h_pre, x[t], sigmoid)
            # 遗忘门
            state['f'][t] = self.new_gate(self.whf, self.wxf, self.bf, h_pre, x[t], sigmoid)
            # 输出门
            state['o'][t] = self.new_gate(self.who, self.wxo, self.bo, h_pre, x[t], sigmoid)
            # 输入状态c~
            state['c_'][t] = self.new_gate(self.whc, self.wxc, self.bc, h_pre, x[t], tanh)
            # 单元状态 ct = ft * c_pre + it * ~ct
            state['c'][t] = state['f'][t] * state['c'][t - 1] + state['i'][t] * state['c_'][t]
            # 隐藏层输出
            state['h'][t] = state['o'][t] * tanh(state['c'][t])
            # 最终输出 yt = softmax(self.wy.dot(ht) + self.by)
            state['y'][t] = softmax(self.wy.dot(state['h'][t]) + self.by)
        return state

    def backward(self, x, y, learning_rate):
        # 初始化所有参数矩阵梯度
        dwhi, dwxi, dbi = self._init_delta_gate()
        dwhf, dwxf, dbf = self._init_delta_gate()
        dwho, dwxo, dbo = self._init_delta_gate()
        dwhc, dwxc, dbc = self._init_delta_gate()
        dwy, dby = np.zeros(self.wy.shape), np.zeros(self.by.shape)

        # 初始化 delta_ct，因为后向传播过程中，此值需要累加
        delta_ct = np.zeros((self.hidden_dim, 1))

        # 前向计算
        state = self.forward(x)

        # 目标函数(CrossEntryLoss)对输出 y 的偏导数
        delta_y = state['y']
        delta_y[np.arange(len(y)), y] -= 1

        for t in np.arange(len(y))[::-1]:
            # 输出层wy, by的偏导数
            dwy += delta_y[t].dot(state['h'][t].reshape(1, -1))
            dby += delta_y[t]

            # 目标函数对隐藏状态的偏导数
            delta_ht = self.wy.T.dot(delta_y[t])

            # 各个门及状态单元的偏导数
            delta_ot = delta_ht * tanh(state['c'][t]) * state['o'][t] * (1 - state['o'][t])
            delta_ct += delta_ht * state['o'][t] * (1 - tanh(state['c'][t]) ** 2)
            delta_it = delta_ct * state['c_'][t] * state['i'][t] * (1 - state['i'][t])
            delta_ft = delta_ct * state['c'][t - 1] * state['f'][t] * (1 - state['f'][t])
            delta_c_t = delta_ct * state['i'][t] * (1 - state['c_'][t] ** 2)

            # 计算各权重矩阵的偏导数
            dwhf, dwxf, dbf = self.delta_grad(dwhf, dwxf, dbf, delta_ft, state['h'][t - 1], x[t])
            dwhi, dwxi, dbi = self.delta_grad(dwhi, dwxi, dbi, delta_it, state['h'][t - 1], x[t])
            dwhc, dwxc, dbc = self.delta_grad(dwhc, dwxc, dbc, delta_c_t, state['h'][t - 1], x[t])
            dwho, dwxo, dbo = self.delta_grad(dwho, dwxo, dbo, delta_ot, state['h'][t - 1], x[t])

            # 更新权重矩阵
            self.whf, self.wxf, self.bf = self.update_wh_wx(learning_rate, self.whf, self.wxf, self.bf, dwhf, dwxf, dbf)
            self.whi, self.wxi, self.bi = self.update_wh_wx(learning_rate, self.whi, self.wxi, self.bi, dwhi, dwxi, dbi)
            self.whc, self.wxa, self.ba = self.update_wh_wx(learning_rate, self.whc, self.wxc, self.bc, dwhc, dwxc, dbc)
            self.who, self.wxo, self.bo = self.update_wh_wx(learning_rate, self.who, self.wxo, self.bo, dwho, dwxo, dbo)
            self.wy, self.by = self.wy - learning_rate * dwy, self.by - learning_rate * dby

    def predict(self, x, hist = {'h': 0, 'c': 0}):
        state = self.forward(x, hist)
        # 取出sortmax后值最大的index
        pre_y = np.argmax(state['y'].reshape(len(x), -1), axis=1)
        hist['h'] = state['h'][len(x) - 1]
        hist['c'] = state['c'][len(x) - 1]
        return pre_y, hist

    # 交叉熵损失函数
    def loss(self, x, y):
        loss_sum = 0
        for i in range(len(y)):
            state = self.forward(x[i])
            # 将目标y[i]作为index,取出pre_y相应位置的值，因为只有目标值为1的部分对交叉熵函数才有效
            pre_yi = state['y'][range(len(y[i])), y[i]]
            loss_sum -= np.sum(np.log(pre_yi))
        # 统计所有y中词的个数, 计算平均损失
        N = np.sum([len(yi) for yi in y])
        ave_loss = loss_sum / N
        return ave_loss