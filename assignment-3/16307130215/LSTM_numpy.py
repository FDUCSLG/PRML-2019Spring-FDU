import numpy as np
from torch.autograd import Variable
import math
import torch
import torch.nn as nn

input_size = 5
hidden_size = 10
learning_rate = 1e-2
beta1 = 0.9
beta2 = 0.999

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(x):
    return x * (1 - x)

def tanh(x):
    return np.tanh(x)

def dtanh(x):
    return 1 - x * x

class Param(object):
    def __init__(self, name, value):
        self.name = name
        self.v = value
        self.d = np.zeros_like(value) 
        self.m = np.zeros_like(value) 
        self.g = np.zeros_like(value)
        self.t = 0

class LSTMCell_numpy():
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        z_size = self.hidden_size+self.input_size
        self.W_f = Param('W_f', np.random.uniform(-stdv,stdv,size=(self.hidden_size, z_size)))
        self.b_f = Param('b_f', np.zeros((self.hidden_size, 1)))
        self.W_i = Param('W_i', np.random.uniform(-stdv,stdv,size=(self.hidden_size, z_size)))
        self.b_i = Param('b_i', np.zeros((self.hidden_size, 1)))
        self.W_C = Param('W_C', np.random.uniform(-stdv,stdv,size=(self.hidden_size, z_size)))
        self.b_C = Param('b_C', np.zeros((self.hidden_size, 1)))
        self.W_o = Param('W_o', np.random.uniform(-stdv,stdv,size=(self.hidden_size, z_size)))
        self.b_o = Param('b_o', np.zeros((self.hidden_size, 1)))
        self.W_v = Param('W_v', np.random.uniform(-stdv,stdv,size=(self.input_size, self.hidden_size)))
        self.b_v = Param('b_v', np.zeros((self.input_size, 1)))

    def get_parameters(self):
        return [self.W_f, self.W_i, self.W_C, self.W_o, self.W_v,
                self.b_f, self.b_i, self.b_C, self.b_o, self.b_v]

    def ones_parameters(self):
        for p in self.get_parameters():
            p.v.fill(1)

    def clear_gradients(self):
        for item in [self.W_f, self.W_i, self.W_C, self.W_o, self.W_v,
                     self.b_f, self.b_i, self.b_C, self.b_o, self.b_v]:
            item.d.fill(0)

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
    
    def backward(self, target, dh_next, dC_next, C_prev,
            z, f, i, C_bar, C, o, h, v, y,):

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

        dz = (np.dot(self.W_f.v.T, df)
            + np.dot(self.W_i.v.T, di)
            + np.dot(self.W_C.v.T, dC_bar)
            + np.dot(self.W_o.v.T, do))
        dh_prev = dz[:self.hidden_size, :]
        dC_prev = f * dC
        return dh_prev, dC_prev

class LSTM_numpy():
    def __init__(self, input_size, hidden_size):
        self.lstm_cell = LSTMCell_numpy(input_size, hidden_size)
    
    def forward_backward(self, inputs, targets, h_prev, C_prev):
        x_s, z_s, f_s, i_s, = {}, {}, {}, {}
        C_bar_s, C_s, o_s, h_s = {}, {}, {}, {}
        v_s, y_s = {}, {}

        h_s[-1] = np.copy(h_prev)
        C_s[-1] = np.copy(C_prev)
        
        loss = 0
        seq_len = inputs.shape[0]
        for t in range(seq_len):
            x_s[t]=np.array(inputs[t]).T
            (z_s[t], f_s[t], i_s[t], C_bar_s[t], C_s[t], o_s[t], h_s[t],v_s[t], y_s[t]) \
            = self.lstm_cell.forward(x_s[t], h_s[t - 1], C_s[t - 1]) 
            loss += -np.log(y_s[t][targets[t], 0])
        
        self.lstm_cell.clear_gradients()

        dh_next = np.zeros_like(h_s[0]) 
        dC_next = np.zeros_like(C_s[0])

        for t in reversed(range(seq_len)):
            dh_next, dC_next = self.lstm_cell.backward(
                                        target = targets[t], dh_next = dh_next,
                                        dC_next = dC_next, C_prev = C_s[t-1],
                                        z = z_s[t], f = f_s[t], i = i_s[t], C_bar = C_bar_s[t],
                                        C = C_s[t], o = o_s[t], h = h_s[t], v = v_s[t],
                                        y = y_s[t])

        for p in self.lstm_cell.get_parameters():
            p.t += 1
            p.m = beta1 * p.m + (1-beta1)*p.d
            p.g = beta2 * p.g + (1-beta2)*p.d*p.d
            mb = p.m/(1-beta1**p.t)
            gb = p.g/(1-beta2**p.t)
            p.v -= learning_rate * mb/(np.sqrt(gb+1e-8))
            # p.m += p.d * p.d
            # p.v += -(learning_rate * p.d / np.sqrt(p.m + 1e-8))

        return loss, h_s[seq_len - 1], C_s[seq_len - 1]

class LSTM_pytorch(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM_pytorch, self).__init__()
        self.hidden_size = hidden_size
        self.weight = torch.nn.Parameter(torch.Tensor(hidden_size * 4,input_size + hidden_size))
        self.bias = torch.nn.Parameter(torch.Tensor(hidden_size * 4,1))
        self.W = torch.nn.Parameter(torch.ones(input_size,hidden_size))
        self.b = torch.nn.Parameter(torch.ones(input_size,1))
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)
    
    def ones_weights(self):
        for p in self.parameters():
            nn.init.ones_(p.data)

    #input：[seq_len,batch,input_size]
    def forward(self, input, hidden = None, cell = None):
        seq_len, batch_size, _ = input.shape
        output = []
        if hidden is None:
            hidden = input.data.new(self.hidden_size,batch_size).fill_(0).float()
            cell = input.data.new(self.hidden_size,batch_size).fill_(0).float()
        HS = self.hidden_size
        for t in range(seq_len):
            z = torch.cat((hidden, input[t].transpose(1,0)))
            gates = self.weight @ z+ self.bias
            f = torch.sigmoid(gates[:HS,:])
            i = torch.sigmoid(gates[HS:HS*2,:])
            C_bar = torch.tanh(gates[HS*2:HS*3,:])
            o = torch.sigmoid(gates[HS*3:,:])
            cell = torch.add(torch.mul(cell, f), torch.mul(C_bar, i))
            hidden = torch.mul(torch.tanh(cell), o)
            output.append(hidden.unsqueeze(0))
            
        output = torch.cat(output, dim=0).transpose(2,1)
        output = output.view(4,-1)
        output = self.W @ output.transpose(1,0) + self.b
        output = output.transpose(1,0)
        pred = torch.nn.functional.softmax(output, dim=1)
        target_onehot = torch.zeros_like(pred)
        target_onehot = target_onehot.scatter(1, target, 1)
        loss = torch.sum(-torch.log(torch.sum(torch.mul(pred, target_onehot), dim =1)))
        return loss


if __name__ == "__main__":
    LSTM_pytorch = LSTM_pytorch(input_size = input_size, hidden_size = hidden_size)
    LSTM_numpy = LSTM_numpy(input_size = input_size, hidden_size = hidden_size)
    LSTM_pytorch.ones_weights()
    LSTM_numpy.lstm_cell.ones_parameters()
    optimizer = torch.optim.Adam(LSTM_pytorch.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    h_prev = np.zeros((hidden_size, 1))
    C_prev = np.zeros((hidden_size, 1))
    for i in range(20):
        input = Variable(torch.randn(4,1,input_size))
        target = Variable(torch.ones(4,1)).long()
        optimizer.zero_grad()
        loss1 = LSTM_pytorch(input)
        loss1.backward()
        optimizer.step()
        loss2, _,_ = LSTM_numpy.forward_backward(input.data, target.data, h_prev, C_prev)
        print((loss1-loss2).data)
        w1 = LSTM_numpy.lstm_cell.W_f.d[0]
        w2 = LSTM_pytorch.weight.grad[0]
        print((w1-w2).data)
    