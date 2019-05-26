import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable

class MyLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, cell_size):
        super(MyLSTMCell, self).__init__()
        self.hidden_size = hidden_size
        self.cell_size = cell_size
        self.gate = nn.Linear(input_size + hidden_size, cell_size)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, input, hidden):
        hx,cx = hidden
        combined = torch.cat((input, hx), 1)
        f_gate = self.gate(combined)
        f_gate = self.sigmoid(f_gate)
        i_gate = self.gate(combined)
        i_gate = self.sigmoid(i_gate)
        o_gate = self.gate(combined)
        o_gate = self.sigmoid(o_gate)
        c_gate = self.gate(combined)
        c_gate = self.tanh(c_gate)
        cx = torch.add(torch.mul(cx, f_gate), torch.mul(c_gate, i_gate))
        hx = torch.mul(self.tanh(cx), o_gate)
        hidden = hx,cx
        return hidden

class MyLSTM(nn.Module):
    def __init__(self,input_size, hidden_size, cell_size):
        super(MyLSTM,self).__init__()
        self.mylstmcell = MyLSTMCell(input_size, hidden_size, cell_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell_size = cell_size
        
    def forward(self,input, hidden = None):
        # hidden is the initial [h_0,c_0]
        seq_len = input.size(0)
        max_batch_size = input.size(1)
        zeros = torch.zeros(max_batch_size, self.hidden_size,dtype=input.dtype, device=input.device)
        if hidden is None:
            hidden = (zeros,zeros)
        hx,cx=hidden
        output = []
        for i in range(seq_len):
            hx,cx = self.mylstmcell(input[i,:,:],(hx,cx))
            output.append(hx)
        hidden = hx,cx
        output2 = torch.cat(output,0)
        
        output = torch.stack(output,0)
        return output,hidden
        
