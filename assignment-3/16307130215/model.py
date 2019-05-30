import torch
import torch.nn as nn
from torch.autograd import Variable

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.f_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.i_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.o_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.C_gate = nn.Linear(input_size + hidden_size, hidden_size)
        # self.f_gate.weight.data.fill_(0)
        # self.i_gate.weight.data.fill_(0)
        # self.o_gate.weight.data.fill_(0)
        # self.C_gate.weight.data.fill_(0)

    #input：[seq_len,batch,input_size]
    def forward(self, input, hidden = None, cell = None):
        # print(self.f_gate.weight.data)
        seq_len, batch_size, _ = input.shape
        output = []
        if hidden is None:
            hidden = input.data.new(batch_size, self.hidden_size).fill_(0).float()
            cell = input.data.new(batch_size, self.hidden_size).fill_(0).float()
        for t in range(seq_len):
            z = torch.cat((hidden, input[t]), 1)
            f = torch.sigmoid(self.f_gate(z))
            i = torch.sigmoid(self.i_gate(z))
            o = torch.sigmoid(self.o_gate(z))
            C_bar = torch.tanh(self.C_gate(z))
            cell = torch.add(torch.mul(cell, f), torch.mul(C_bar, i))
            hidden = torch.mul(torch.tanh(cell), o)
            output.append(hidden.unsqueeze(0))
        
        output = torch.cat(output, dim=0)
        return output, hidden, cell

class PoetryModel(nn.Module):

    def __init__(self, vocab_size, embedding_size, hidden_size):
        super(PoetryModel,self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.lstm = LSTM(embedding_size, hidden_size)
        self.transfrom = nn.Linear(hidden_size, vocab_size)
        # self.embeddings.weight.data.fill_(0)
        # self.transfrom.weight.data.fill_(0)

    #input：[seq_len,batch]
    def forward(self, input, hidden = None, ceil = None):
        input = input.transpose(1,0) #input：[batch,seq_len]->[seq_len,batch]
        seq_len, batch_size = input.shape
        if hidden is None:
            hidden = input.data.new(batch_size, self.hidden_size).fill_(0).float()
            ceil = input.data.new(batch_size, self.hidden_size).fill_(0).float()

        embeds = self.embeddings(input)
        output, hidden, ceil = self.lstm(embeds, hidden, ceil)
        output = output.transpose(1,0).contiguous()
        output = self.transfrom(output.view(seq_len*batch_size, -1))
        return {"output":output, "hidden":hidden, "ceil":ceil}
