import numpy as np

def sigmoid(x):
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)

class Embedding(object):
    def __init__(self):
        self.cache = None

    def forward(self, x, W):
        out = W[x]
        self.cache = (x, W)
        return out

    def backward(self, dout):
        x, W = self.cache
        dW = np.zeros(W.shape)
        np.add.at(dW, x, dout)
        return dW

class LSTM(object):
    
    def __init__(self):
        self.cache = None

    def step_forward(self, x, prev_h, prev_c, Wx, Wh, b):
        N, H = prev_h.shape

        a = np.dot(x, Wx) + np.dot(prev_h, Wh) + b
        ai, af, ao, ag = a[:, 0:H], a[:, H:2*H], a[:, 2*H:3*H], a[:, 3*H:4*H]
        i_t, f_t, o_t, g_t = sigmoid(ai), sigmoid(af), sigmoid(ao), np.tanh(ag)

        next_c = f_t * prev_c + i_t * g_t
        next_h = o_t * np.tanh(next_c)

        cache = (i_t, f_t, o_t, g_t, next_c, next_h, x, prev_h, prev_c, Wx, Wh, b)

        return next_h, next_c, cache

    def step_backward(self, dnext_h, dnext_c, cache):
        i_t, f_t, o_t, g_t, next_c, next_h, x, prev_h, prev_c, Wx, Wh, b = cache

        do_t = np.tanh(next_c) * dnext_h
        dc_t = o_t * (1 - np.tanh(next_c) * np.tanh(next_c)) * dnext_h + dnext_c
        df_t = prev_c * dc_t
        di_t = g_t * dc_t
        dprev_c = f_t * dc_t
        dg_t = i_t * dc_t

        dai = (1 - i_t) * i_t * di_t
        daf = (1 - f_t) * f_t * df_t
        dao = (1 - o_t) * o_t * do_t
        dag = (1 - g_t * g_t) * dg_t
        da = np.hstack((dai, daf, dao, dag))

        dWx = np.dot(x.T, da)
        dWh = np.dot(prev_h.T, da)
        db = np.sum(da, axis=0)
        dx = np.dot(da, Wx.T)
        dprev_h = np.dot(da, Wh.T)

        return dx, dprev_h, dprev_c, dWx, dWh, db

    def forward(self, x, h0, Wx, Wh, b):
        N, T, D = x.shape
        H = b.shape[0] // 4
        h = np.zeros((N, T, H))
        self.cache = {}
        prev_h = h0
        prev_c = np.zeros((N, H))

        for i in range(T):
            x_t = x[:, i, :]
            next_h, next_c, self.cache[i] = self.step_forward(x_t, prev_h, prev_c, Wx, Wh, b)
            prev_h = next_h
            prev_c = next_c
            h[:, i, :] = prev_h

        return h, prev_h

    def backward(self, dh):
        N, T, H = dh.shape
        i_t, f_t, o_t, g_t, next_c, next_h, x, prev_h, prev_c, Wx, Wh, b = self.cache[T-1]
        D = x.shape[1]

        dprev_h = np.zeros((N, H))
        dprev_c = np.zeros((N, H))
        dx = np.zeros((N, T, D))
        dh0 = np.zeros((N, H))
        dWx = np.zeros((D, 4*H))
        dWh = np.zeros((H, 4*H))
        db = np.zeros((4*H, ))

        for i in range(T-1, -1, -1):
            step_cache = self.cache[i]
            dnext_h = dh[:, i, :] + dprev_h
            dnext_c = dprev_c
            dx[:, i, :], dprev_h, dprev_c, dWx_t, dWh_t, db_t = \
                self.step_backward(dnext_h, dnext_c, step_cache)
            dWx += dWx_t
            dWh += dWh_t
            db += db_t

        dh0 = dprev_h

        return dx, dh0, dWx, dWh, db

class TemporalFC(object):
    def __init__(self):
        self.cache = None

    def forward(self, x, w, b):
        N, T, D = x.shape
        M = b.shape[0]
        out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
        self.cache = x, w, b, out
        return out

    def backward(self, dout):
        x, w, b, out = self.cache
        N, T, D = x.shape
        M = b.shape[0]

        dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
        dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
        db = dout.sum(axis=(0, 1))

        return dx, dw, db

class FC(object):
    def __init__(self):
        self.cache = None

    def forward(self, x, w, b):
        out = x.reshape(x.shape[0], -1).dot(w) + b
        self.cache = (x, w, b)
        return out

    def backward(self, dout):
        x, w, b = self.cache
        dx = dout.dot(w.T).reshape(x.shape)
        dw = x.reshape(x.shape[0], -1).T.dot(dout)
        db = np.sum(dout, axis=0)
        return dx, dw, db
