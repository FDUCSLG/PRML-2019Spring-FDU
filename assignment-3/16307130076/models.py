import torch
import torch.nn as nn
import math
from torch.autograd import Variable


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, cell_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell_size = cell_size
        self.fgate = nn.Linear(input_size + hidden_size, cell_size)
        self.igate = nn.Linear(input_size + hidden_size, cell_size)
        self.cgate = nn.Linear(input_size + hidden_size, cell_size)
        self.ogate = nn.Linear(input_size + hidden_size, cell_size)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def init_weights(self, m):
        stdv = 1.0 / math.sqrt(self.input_size)
        if isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight.data, -stdv, stdv)

    def forward(self, input, hidden, cell):
        combined = torch.cat((input, hidden), 1)
        f_gate = self.fgate(combined)
        i_gate = self.igate(combined)
        c_gate = self.cgate(combined)
        o_gate = self.ogate(combined)
        f_gate = self.sigmoid(f_gate)
        i_gate = self.sigmoid(i_gate)
        o_gate = self.sigmoid(o_gate)
        c_helper = self.tanh(c_gate)
        cell = torch.add(torch.mul(cell, f_gate), torch.mul(c_helper, i_gate))
        hidden = torch.mul(self.tanh(cell), o_gate)
        return hidden, cell

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, cell_size):
        super(LSTM, self).__init__()
        self.LSTMCell = LSTMCell(
            input_size, hidden_size, cell_size)
        self.LSTMCell.apply(self.LSTMCell.init_weights)

    def forward(self, input, hidden, cell):
        hiddens = None
        steps = range(input.size()[0])
        for i in steps:
            hidden, cell = self.LSTMCell(input[i], hidden, cell)
            hidden_ = hidden.unsqueeze(0)
            if hiddens is None:
                hiddens = hidden_
            else:
                hiddens = torch.cat((hiddens, hidden_), 0)
        return hiddens


class TangPoemGenerator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TangPoemGenerator, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embeds = nn.Embedding(self.output_size, input_size)
        self.lstm = LSTM(input_size, hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        stdv = 1.0 / math.sqrt(self.hidden_size)
        nn.init.uniform_(self.embeds.weight.data, -stdv, stdv)
        nn.init.uniform_(self.output.weight.data, -stdv, stdv)

    def forward(self, x):
        seq_len, batch_size = x.size()
        emb = self.embeds(x)
        hiddens = self.lstm(emb, torch.zeros((x.size()[1],self.hidden_size)).cuda(
        ), torch.zeros((x.size()[1], self.hidden_size)).cuda())
        outputs = self.output(hiddens.view(seq_len * batch_size, -1))
        return outputs


if __name__ == '__main__':
    inputs = Variable(torch.randint(low=0, high=10000, size=(5, 4))).cuda()
    net = TangPoemGenerator(10, 10, 10000).cuda()
    print(net)
    out = net(inputs)
    print(out.size())
    output = torch.max(out, dim=1)
    print(out.size(), output)
