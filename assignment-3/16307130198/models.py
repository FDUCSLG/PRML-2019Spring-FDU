import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter

import numpy as np
import typing
from typing import Optional
from typing import Tuple

from enum import IntEnum
import utils

class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.weight_ih = torch.nn.Parameter(torch.Tensor(input_size, hidden_size * 4))
        self.weight_hh = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size * 4))
        self.bias = Parameter(torch.Tensor(hidden_size * 4))
        self.init_weights()
    
    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)
    
    def forward(self, x: torch.Tensor,
                init_states: Optional[Tuple[torch.Tensor]]=None
               ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.size()
        hidden_seq = []
        if init_states is None:
            h_t, c_t = (torch.zeros(self.hidden_size).to(x.device), 
                        torch.zeros(self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states
        
        HS = self.hidden_size
        for t in range(seq_sz):
            x_t = x[:, t, :]
            
            gates = x_t @ self.weight_ih + h_t @ self.weight_hh + self.bias

            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]),
                torch.sigmoid(gates[:, HS:HS*2]),
                torch.tanh(gates[:, HS*2:HS*3]),
                torch.sigmoid(gates[:, HS*3:]),
            )
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(Dim.batch))

        hidden_seq = torch.cat(hidden_seq, dim=Dim.batch)
        hidden_seq = hidden_seq.transpose(Dim.seq, Dim.batch).contiguous()
        return hidden_seq, (h_t, c_t)

class PoetryModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(PoetryModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = LSTM(embedding_dim, self.hidden_dim, num_layers=1)
        self.linear1 = nn.Linear(self.hidden_dim, vocab_size)
    
    def forward(self, input_data, hidden=None):
        batch_size, seq_len = input_data.shape
        #seq_len, batch_size = input_data.shape
        if hidden is None:
            #  h_0 = 0.01*torch.Tensor(2, batch_size, self.hidden_dim).normal_().cuda()
            #  c_0 = 0.01*torch.Tensor(2, batch_size, self.hidden_dim).normal_().cuda()
            h_0 = input_data.data.new(batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input_data.data.new(batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden
        # size: (seq_len,batch_size,embeding_dim)
        embeds = self.embeddings(input_data)
        # output size: (seq_len,batch_size,hidden_dim)
        output, hidden = self.lstm(embeds, (h_0, c_0))

        # size: (seq_len*batch_size,vocab_size)
        output = self.linear1(output.view(seq_len * batch_size, -1))
        return {"output": output, "hidden": hidden}

# numpy version lstm model
class Param(object):
    def __init__(self, name, value):
        self.name = name
        self.v = value
        self.d = np.zeros_like(value) # derivative
        self.m = np.zeros_like(value) # momentum

class npLSTM(object):
    def __init__(self, input_size, H_size, learning_rate=0.001):
        z_size = H_size + input_size 
        self.input_size = input_size
        self.h_size = H_size
        self.learning_rate=learning_rate
        
        self.W_f = Param('W_f',
                         np.random.randn(H_size, z_size))
        self.b_f = Param('b_f',
                         np.zeros((H_size, 1)))

        self.W_i = Param('W_i',
                         np.random.randn(H_size, z_size))
        self.b_i = Param('b_i',
                         np.zeros((H_size, 1)))
        
        self.W_C = Param('W_C',
                         np.random.randn(H_size, z_size))
        self.b_C = Param('b_C',
                         np.zeros((H_size, 1)))

        self.W_o = Param('W_o',
                         np.random.randn(H_size, z_size))
        self.b_o = Param('b_o',
                         np.zeros((H_size, 1)))
        
        #For final layer to predict the next character
        self.W_v = Param('W_v',
                         np.random.randn(input_size, H_size))
        self.b_v = Param('b_v',
                         np.zeros((input_size, 1)))
    
    def all(self):
        return [self.W_f, self.W_i, self.W_C, self.W_o, self.W_v,
                self.b_f, self.b_i, self.b_C, self.b_o, self.b_v]

    def forward(self, input_data, h_prev, C_prev):
        z = np.row_stack((h_prev, input_data))
        f = utils.sigmoid(np.dot(self.W_f.v, z) + self.b_f.v)
        i = utils.sigmoid(np.dot(self.W_i.v, z) + self.b_i.v)
        C_bar = utils.tanh(np.dot(self.W_C.v, z) + self.b_C.v)

        C = f * C_prev + i * C_bar
        o = utils.sigmoid(np.dot(self.W_o.v, z) + self.b_o.v)
        h = o * utils.tanh(C)

        v = np.dot(self.W_v.v, h) + self.b_v.v
        y = np.exp(v) / np.sum(np.exp(v)) #softmax
        return z, f, i, C_bar, C, o, h, v, y

    def backward(self, target, dh_next, dC_next, C_prev,
                 z, f, i, C_bar, C, o, h, v, y):
        # the following code still needs to be modified.
        # for example: p -> self
        dv = np.copy(y)
        dv[target] -= 1

        self.W_v.d += np.dot(dv, h.T)
        self.b_v.d += dv

        dh = np.dot(self.W_v.v.T, dv)      
        dh += dh_next
        do = dh * utils.tanh(C)
        do = utils.dsigmoid(o) * do
        self.W_o.d += np.dot(do, z.T)
        self.b_o.d += do

        dC = np.copy(dC_next)
        dC += dh * o * utils.dtanh(utils.tanh(C))
        dC_bar = dC * i
        dC_bar = utils.dtanh(C_bar) * dC_bar
        self.W_C.d += np.dot(dC_bar, z.T)
        self.b_C.d += dC_bar

        di = dC * C_bar
        di = utils.dsigmoid(i) * di
        self.W_i.d += np.dot(di, z.T)
        self.b_i.d += di

        df = dC * C_prev
        df = utils.dsigmoid(f) * df
        self.W_f.d += np.dot(df, z.T)
        self.b_f.d += df

        dz = (np.dot(self.W_f.v.T, df)
              + np.dot(self.W_i.v.T, di)
              + np.dot(self.W_C.v.T, dC_bar)
              + np.dot(self.W_o.v.T, do))
        dh_prev = dz[:self.h_size, :]
        dC_prev = f * dC

        return dh_prev, dC_prev

    def clear_gradients(self):
        for p in self.all():
            p.d.fill(0)

    def clip_gradients(self):
        for p in self.all():
            np.clip(p.d, -1, 1, out=p.d)
        
    def forward_backward(self, inputs, targets, h_prev, C_prev):
        # To store the values for each time step
        x_s, z_s, f_s, i_s,  = {}, {}, {}, {}
        C_bar_s, C_s, o_s, h_s = {}, {}, {}, {}
        v_s, y_s =  {}, {}

        # Values at t - 1
        h_s[-1] = np.copy(h_prev)
        C_s[-1] = np.copy(C_prev)
        
        loss = 0

        # forward -> loop through the time step
        T_steps = inputs.shape[1]
        for t in range(T_steps):
            x_s[t] = np.zeros((self.input_size, 1))
            x_s[t] = np.expand_dims(inputs[:,t], 1)

            (z_s[t], f_s[t], i_s[t],
             C_bar_s[t], C_s[t], o_s[t], h_s[t],
             v_s[t], y_s[t]) = \
                    self.forward(x_s[t], h_s[t - 1], C_s[t - 1])

            loss += -np.log(y_s[t][targets[t], 0])
        
        self.clear_gradients()

        dh_next = np.zeros_like(h_s[0]) #dh from the next character
        dC_next = np.zeros_like(C_s[0]) #dh from the next character
        
        # backward -> loop reversely through the time step
        for t in reversed(range(T_steps)):
            dh_next, dC_next = \
                    self.backward(target = targets[t], dh_next = dh_next,
                             dC_next = dC_next, C_prev = C_s[t-1],
                             z = z_s[t], f = f_s[t], i = i_s[t], C_bar = C_bar_s[t],
                             C = C_s[t], o = o_s[t], h = h_s[t], v = v_s[t],
                             y = y_s[t])

        # clip the gradient, preventing gradient explosion
        # self.clip_gradients() 

        return loss, h_s[T_steps - 1], C_s[T_steps - 1]
        
    def update_parameters(self):
        # Adagrad update form
        for p in self.all():
            p.m += p.d * p.d # Calculate sum of gradients
            p.v += -(self.learning_rate * p.d / np.sqrt(p.m + 1e-8))

