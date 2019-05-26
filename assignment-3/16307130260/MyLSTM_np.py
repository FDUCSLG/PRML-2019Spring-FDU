import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))



def LSTMCellForward(x, prev_h, prev_c, Wx, Wh, b):
    """
        Batch size N
        input dim D
        hidden dim H

        x (N, D)
        prev_h (N, H)
        prev_c (N, H)
        Wx (D, 4H)
        Wh (H, 4H)
        b (4H)

        next_h (N, H)
        next_c (N, H)
        cache (x, prev_h, prev_c, Wx, Wh, i_t, f_t, o_t, g_t, next_h, next_c)
    """
    H = prev_h.shape[1]

    gates = x @ Wx + prev_h @ Wh + b

    i_t = sigmoid(gates[:, 0: H])
    f_t = sigmoid(gates[:, H: 2*H])
    g_t = sigmoid(gates[:, 2*H: 3*H])
    o_t = np.tanh(gates[:, 3*H: 4*H])

    next_c = f_t * prev_c + i_t * g_t
    next_h = o_t * np.tanh(next_c)

    cache = (x, prev_h, prev_c, Wx, Wh, i_t, f_t, o_t, g_t, next_h, next_c)
    return next_h, next_c, cache

def LSTMCellBackward(dnext_h, dnext_c, cache):
    """
        Batch size N
        hidden dim H

        dnext_h (N, H)
        dnext_c (N, H)
        cache (x, prev_h, prev_c, Wx, Wh, i_t, f_t, o_t, g_t, next_h, next_c)
    """
    (x,prev_h, prev_c, Wx, Wh, i_t, f_t, o_t, g_t, next_h, next_c) = cache

    dnext_c = dnext_c + o_t * (1 - np.tanh(next_c)**2) * dnext_h

    di_t = dnext_c * g_t
    df_t = dnext_c * prev_c 
    do_t = dnext_h * np.tanh(next_c)
    dg_t = dnext_c * i_t

    dprev_c = f_t * dnext_c
    dgates = np.hstack((i_t * (1 - i_t) * di_t, f_t * (1 - f_t) * df_t,
                        o_t * (1 - o_t) * do_t, (1 - g_t**2) * dg_t))
    dx = dgates @ Wx.T
    dprev_h = dgates @ Wh.T
    dWx = x.T @ dgates
    dWh = prev_h.T @ dgates
    db = np.sum(dgates, axis=0)
    return dx, dprev_h, dprev_c, dWx, dWh, db

def LSTMForward(x, h0, Wx, Wh, b):
    """
    Batch size N
    Seq length T
    input dim D
    hidden dim H

    x (N, T, D)
    h0 (N, H)
    Wx (D, 4H)
    Wh (H, 4H)
    b (4H)
    """
    N, T, D = x.shape
    _, H = h0.shape
    h = np.zeros((N,T,H))
    c = np.zeros((N,T,H))
    c0 = np.zeros((N,H))
    cache = {}
    for t in range(T):
        if t==0:
            h[:,t,:], c[:,t,:], cache[t] = LSTMCellForward(x[:,t,:], h0, c0, Wx, Wh, b)
        else:
            h[:,t,:], c[:,t,:], cache[t] = LSTMCellForward(
                x[:,t,:], h[:,t-1,:], c[:,t-1,:], Wx, Wh, b)
    return h, cache


def LSTMBackward(dh, cache):
    """
    Batch size N
    Seq length T
    hidden dim H

    dh (N, T, H)
    """
    (N, T, H) = dh.shape
    x,prev_h, prev_c, Wx, Wh, _, _, _, _, _, _= cache[T-1]
    D = x.shape[1]

    dx = np.zeros((N,T,D))
    dWx = np.zeros(Wx.shape)
    dWh = np.zeros(Wh.shape)
    db = np.zeros((4*H))
    dprev = np.zeros(prev_h.shape)
    dprev_c = np.zeros(prev_c.shape)

    for t in range(T-1,-1,-1):
        dx[:,t,:], dprev, dprev_c, dWx_local, dWh_local, db_local = LSTMCellBackward(
            dh[:,t,:] + dprev, dprev_c, cache[t])
        dWx += dWx_local
        dWh += dWh_local
        db += db_local
    dh0 = dprev
    return dx, dh0, dWx, dWh, db

class MyLSTM_np:
    def __init__(self, embedding_dim, hidden_dim):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.Wx = np.random.normal(0.0, 0.1, (embedding_dim, 4 * hidden_dim))
        self.Wh = np.random.normal(0.0, 0.1, (hidden_dim, 4 * hidden_dim))
        self.b = np.zeros(4 * hidden_dim)

    def forward(self, x, h0=None):
        if h0 == None:
            h0 = np.zeros((x.shape[0], self.hidden_dim))
        h, self.cache =  LSTMForward(x, h0, self.Wx, self.Wh, self.b)
        return h

    
    def backward(self, dh):
        dx, dh0, dWx, dWh, db = LSTMBackward(dh, self.cache)
        return dx, dh0, dWx, dWh, db

def test():
    embedding_dim = 4
    hidden_dim = 8

    lstm = nn.LSTM(embedding_dim, hidden_dim, bias=False, num_layers=1)
    mylstm = MyLSTM_np(embedding_dim, hidden_dim)

    params = list(lstm.parameters())

    # reorgnize for weight
    wx_temp = params[0].detach().numpy().T
    mylstm.Wx[:, :2 * hidden_dim] = wx_temp[:,:2 * hidden_dim]
    mylstm.Wx[:, 2 * hidden_dim:3 * hidden_dim] = wx_temp[:,3 * hidden_dim:4 * hidden_dim]
    mylstm.Wx[:, 3 * hidden_dim:4 * hidden_dim] = wx_temp[:,2 * hidden_dim:3 * hidden_dim]
    Wh_temp = params[1].detach().numpy().T
    mylstm.Wh[:, :2 * hidden_dim] = Wh_temp[:,:2 * hidden_dim]
    mylstm.Wh[:, 2 * hidden_dim:3 * hidden_dim] = Wh_temp[:,3 * hidden_dim:4 * hidden_dim]
    mylstm.Wh[:, 3 * hidden_dim:4 * hidden_dim] = Wh_temp[:,2 * hidden_dim:3 * hidden_dim]

    xn = np.random.random((1, 1, 4))
    xt = Variable(torch.from_numpy(xn.astype(np.float32)))
    
    hn = mylstm.forward(xn)
    ht = lstm(xt)
    loss = torch.sum(ht[0])
    print(hn-ht[0].detach().numpy())

    dh = np.ones((1, 1, hidden_dim))
    dx, _, dWx, dWh, _ = mylstm.backward(dh)
    loss.backward()
    
    dWh[:, :2 * hidden_dim] -= params[1].grad.numpy().T[:,:2 * hidden_dim]
    dWh[:, 2 * hidden_dim:3 * hidden_dim] -= params[1].grad.numpy().T[:,3 * hidden_dim:4 * hidden_dim]
    dWh[:, 3 * hidden_dim:4 * hidden_dim] -= params[1].grad.numpy().T[:,2 * hidden_dim:3 * hidden_dim]
    params = list(lstm.parameters())
    
    dWx[:, :2 * hidden_dim] -= params[0].grad.numpy().T[:,:2 * hidden_dim]
    dWx[:, 2 * hidden_dim:3 * hidden_dim] -= params[0].grad.numpy().T[:,3 * hidden_dim:4 * hidden_dim]
    dWx[:, 3 * hidden_dim:4 * hidden_dim] -= params[0].grad.numpy().T[:,2 * hidden_dim:3 * hidden_dim]
    print(dWx)


    """
        i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{(t-1)} + b_{hi}) \\
        f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{(t-1)} + b_{hf}) \\
        g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{(t-1)} + b_{hg}) \\
        o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{(t-1)} + b_{ho}) \\
    """


if __name__ == "__main__":
    test()