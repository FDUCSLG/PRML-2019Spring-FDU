import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.f_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.i_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.o_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.C_gate = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, input, hidden=None, cell=None):
        seq_len, batch_size, _ = input.shape
        output = []
        if hidden is None:
            hidden = input.data.new(batch_size, self.hidden_size).fill_(0).float()
            cell = input.data.new(batch_size, self.hidden_size).fill_(0).float()
        for j in range(seq_len):
            z = torch.cat((hidden, input[j]), 1)
            f = torch.sigmoid(self.f_gate(z))
            i = torch.sigmoid(self.i_gate(z))
            o = torch.sigmoid(self.o_gate(z))
            C_hat = torch.tanh(self.C_gate(z))
            cell = torch.add(torch.mul(cell, f), torch.mul(C_hat, i))
            hidden = torch.mul(torch.tanh(cell), o)
            output.append(hidden.unsqueeze(0))

        output = torch.cat(output, dim=0)
        return output, hidden, cell


class PoetryModel(nn.Module):
    def __init__(self, vocabulary_size, embedding_size, hidden_size):
        super(PoetryModel,self).__init__()
        self.vocabulary_size = vocabulary_size
        self.hidden_size = hidden_size
        self.embeddings = nn.Embedding(vocabulary_size, embedding_size)
        self.lstm = LSTM(embedding_size, hidden_size)
        self.linear = nn.Linear(hidden_size, vocabulary_size)

    def forward(self, input, hidden=None, cell=None):
        input = input.transpose(1,0)
        seq_len, batch_size = input.shape
        if hidden is None:
            hidden = input.data.new(batch_size, self.hidden_size).fill_(0).float()
            cell = input.data.new(batch_size, self.hidden_size).fill_(0).float()

        embeds = self.embeddings(input)
        output, hidden, cell = self.lstm(embeds, hidden, cell)
        output = output.transpose(1, 0).contiguous()
        output = self.linear(output.view(seq_len*batch_size, -1))
        return {"output": output, "hidden": hidden, "cell": cell}
