import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math

class PoetryModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(PoetryModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # self.embeddings.weight.data.fill_(0)
        # self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=1)
        self.lstm = MyLSTM(embedding_dim, self.hidden_dim)
        self.linear1 = nn.Linear(self.hidden_dim, vocab_size)
        # self.linear1.weight.data.fill_(0.00001)

    def forward(self, input, hidden=None):
        seq_len, batch_size = input.size()
        # print("embeddings.weight")
        # print(self.embeddings.weight)
        if hidden is None:
            # h_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
            # c_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
            h_0 = input.data.new(batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden
        # size: (seq_len,batch_size,embeding_dim)
        embeds = self.embeddings(input)
        # output size: (seq_len,batch_size,hidden_dim)
        output, hidden = self.lstm(embeds, (h_0, c_0))

        # size: (seq_len*batch_size,vocab_size)
        output = self.linear1(output.view(seq_len * batch_size, -1))
        return output, hidden


class MyLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_dim
        self.weight_ih = Parameter(torch.Tensor(input_dim, hidden_dim * 4))
        self.weight_hh = Parameter(torch.Tensor(hidden_dim, hidden_dim * 4))
        self.bias = Parameter(torch.Tensor(hidden_dim * 4))
        self.init_weights()
    
    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
                # nn.init.xavier_normal_(p.data)
                # fan_in, _ = nn.init._calculate_fan_in_and_fan_out(p.data)
                # nn.init.normal_(p.data, 0, math.sqrt(1/fan_in))
                # nn.init.zeros_(p.data)
            else:
                nn.init.zeros_(p.data)
        
    def forward(self, x, hidden=None):
        # print("weight_ih")
        # print(self.weight_ih)
        """
            input dim: (seq, batch, input_dim)
            output dim: (seq, batch, hidden)
        """
        seq_size, batch_size, _ = x.size()
        if hidden is None:
            h_t, c_t = (torch.zeros(self.hidden_size).to(x.device), 
                        torch.zeros(self.hidden_size).to(x.device))
        else:
            (h_t, c_t) = hidden
        hidden_seq = []
        hidden_size = self.hidden_size
        for t in range(seq_size):
            x_t = x[t, :, :]
            gates = x_t @ self.weight_ih + h_t @ self.weight_hh + self.bias
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :hidden_size]), # input
                torch.sigmoid(gates[:, hidden_size:hidden_size*2]), # forget
                torch.tanh(gates[:, hidden_size*2:hidden_size*3]),
                torch.sigmoid(gates[:, hidden_size*3:hidden_size*4]), # output
            )
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)

        hidden_seq = hidden_seq.contiguous().view(seq_size, batch_size, -1)
        return hidden_seq, (h_t, c_t)