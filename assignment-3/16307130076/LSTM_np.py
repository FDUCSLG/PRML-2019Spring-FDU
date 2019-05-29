import numpy as np


def softmax(x):
    x = np.array(x)
    max_x = np.max(x)
    return np.exp(x-max_x) / np.sum(np.exp(x-max_x))


def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))


def tanh(x):
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))


class npLSTM:
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.whi, self.wxi, self.bi = self._init_weight()
        self.whf, self.wxf, self.bf = self._init_weight()
        self.who, self.wxo, self.bo = self._init_weight()
        self.wha, self.wxa, self.ba = self._init_weight()

        self.wy, self.by = np.random.uniform(-np.sqrt(1.0/self.hidden_dim), np.sqrt(1.0/self.hidden_dim), (self.data_dim, self.hidden_dim)
                                             ), np.random.uniform(-np.sqrt(1.0/self.hidden_dim), np.sqrt(1.0/self.hidden_dim), (self.data_dim, 1))

    def _init_weight(self):
        wh = np.random.uniform(-np.sqrt(1.0/self.hidden_dim), np.sqrt(1.0/self.hidden_dim),
                               (self.hidden_dim, self.hidden_dim))
        wx = np.random.uniform(-np.sqrt(1.0/self.input_dim), np.sqrt(1.0/self.input_dim),
                               (self.hidden_dim, self.input_dim))
        b = np.random.uniform(-np.sqrt(1.0/self.input_dim), np.sqrt(1.0/self.input_dim),
                              (self.hidden_dim, 1))
        return wh, wx, b

    def _init_state(self, T):
        iss = np.array([np.zeros((self.hidden_dim, 1))] * (T + 1))
        fss = np.array([np.zeros((self.hidden_dim, 1))] * (T + 1))
        oss = np.array([np.zeros((self.hidden_dim, 1))] * (T + 1))
        ass = np.array([np.zeros((self.hidden_dim, 1))] * (T + 1))
        hss = np.array([np.zeros((self.hidden_dim, 1))] * (T + 1))
        css = np.array([np.zeros((self.hidden_dim, 1))] * (T + 1))
        ys = np.array([np.zeros((self.hidden_dim, 1))] * T)

        return {
            'iss': iss,
            'fss': fss,
            'oss': oss,
            'ass': ass,
            'hss': hss,
            'css': css,
            'ys': ys
        }

        def _cal_gate(self, wh, wx, b, ht_pre, x, fn):
            return fn(wh.dot(ht_pre) + wx.dot(x) + b)

        def forward(self, x):
            T = len(x)
            stats = self._init_state(T)

            for t in range(T):
                ht_pre = np.array(stats['hss'][t - 1]).reshape(-1, 1)
                stats['iss'][t] = self._cal_gate(
                    self.whi, self.wxi, self.bi, ht_pre, x[t], sigmoid)
                stats['fss'][t] = self._cal_gate(
                    self.whf, self.wxf, self.bf, ht_pre, x[t], sigmoid)
                stats['oss'][t] = self._cal_gate(
                    self.who, self.wxo, self.bo, ht_pre, x[t], sigmoid)
                stats['ass'][t] = self._cal_gate(
                    self.wha, self.wxa, self.ba, ht_pre, x[t], tanh)
                stats['css'][t] = stats['fss'][t] * \
                    stats['css'][t - 1] + stats['iss'][t] * stats['ass'[t]]
                stats['hss'][t] = stats['oss'][t] * tanh(stats['css'][t])
                stats['ys'][t] = softmax(
                    self.wy.dot(stats['hss'][t]) + self.by)

            return stats

        def predict(self, x):
            stats = self.forward(x)
            pre_y = np.argmax(stats['ys'].reshape(len(x), -1), axis=1)
            return pre_y

        def loss(self, x, y):
            cost = 0
            for i in range(len(y)):
                stats = self.forward(x[i])
                pre_yi = stats['ys'][range(len(y[i])), y[i]]
                cost -= np.sum(np.log(pre_yi))

            N = np.sum([len(yi) for yi in y])
            avg_loss = cost / N

            return avg_loss

        def _init_grad(self):
            dwh = np.zeros(self.whi.shape)
            dwx = np.zeros(self.wxi.shape)
            db = np.zeros(self.bi.shape)

            return dwh, dwx, db

        def _cal_grad_delta(self, dwh, dwx, db, delta_net, ht_pre, x):
            dwh += delta_net * ht_pre
            dwx += delta_net * x
            db += delta_net
            return dwh, dwx, db

        def bptt(self, x, y):
            dwhi, dwxi, dbi = self._init_grad()
            dwhf, dwxf, dbf = self._init_grad()
            dwho, dwxo, dbo = self._init_grad()
            dwha, dwxa, dba = self._init_grad()
            dwy, dby = np.zeros(self.wy.shape), np.zeros(self.by.shape)

            delta_ct = np.zeros((self.hidden_dim, 1))

            stats = self.forward(x)
            delta_o = stats['ys']
            delta_o[np.arange(len(y)), y] -= 1

            for t in np.arange(len(y))[::-1]:
                dwy += delta_o[t].dot(stats['hss'][t].reshape(1, -1))
                dby += delta_o[t]

                delta_ht = self.wy.T.dot(delta_o[t])
                delta_ot = delta_ht * tanh(stats['css'][t])
                delta_ct += delta_ht * \
                    stats['oss'][t] * (1-tanh(stats['css'][t])**2)
                delta_it = delta_ct * stats['ass'][t]
                delta_ft = delta_ct * stats['css'][t-1]
                delta_at = delta_ct * stats['iss'][t]

                delta_at_net = delta_at * (1-stats['ass'][t]**2)
                delta_it_net = delta_it * stats['iss'][t] * (1-stats['iss'][t])
                delta_ft_net = delta_ft * stats['fss'][t] * (1-stats['fss'][t])
                delta_ot_net = delta_ot * stats['oss'][t] * (1-stats['oss'][t])

                dwhf, dwxf, dbf = self._cal_grad_delta(
                    dwhf, dwxf, dbf, delta_ft_net, stats['hss'][t-1], x[t])
                dwhi, dwxi, dbi = self._cal_grad_delta(
                    dwhi, dwxi, dbi, delta_it_net, stats['hss'][t-1], x[t])
                dwha, dwxa, dba = self._cal_grad_delta(
                    dwha, dwxa, dba, delta_at_net, stats['hss'][t-1], x[t])
                dwho, dwxo, dbo = self._cal_grad_delta(
                    dwho, dwxo, dbo, delta_ot_net, stats['hss'][t-1], x[t])

            return [dwhf, dwxf, dbf, dwhi, dwxi, dbi, dwha, dwxa, dba, dwho, dwxo, dbo, dwy, dby]

        def _update_weight(self, lr, wh, wx, b, dwh, dwx, db):
            wh -= lr * dwh
            wx -= lr * dwx
            b -= lr * db

        def optimize(self, x, y, lr):
            dwhf, dwxf, dbf, dwhi, dwxi, dbi, dwha, dwxa, dba, dwho, dwxo, dbo, dwy, dby = self.bptt(
                x, y)

            self.whf, self.wxf, self, bf = self, _update_weight(
                lr, self.whf, self.wxf, self.bf, dwhf, dwxf, dbf)
            self.whi, self.wxi, self, bi = self, _update_weight(
                lr, self.whi, self.wxi, self.bi, dwhi, dwxi, dbi)
            self.wha, self.wxa, self, ba = self, _update_weight(
                lr, self.wha, self.wxa, self.ba, dwha, dwxa, dba)
            self.who, self.wxo, self, bo = self, _update_weight(
                lr, self.who, self.wxo, self.bo, dwho, dwxo, dbo)

            self.wy, self.by = self.wy - lr * dwy, self.by - lr * self.dby

        def train(self, x_train, y_train, lr, n_epoch):
            losses = []

            for epoch in range(n_epoch):
                for i in range(len(y_train)):
                    self.optimize(x_train[i], y_train[i], lr)

                loss = self.loss(x_train, y_train)
                losses.append(loss)
