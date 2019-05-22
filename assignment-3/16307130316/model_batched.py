import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter

class LSTM_torch(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LSTM_torch, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.hidden_dim = hidden_dim
        self.weight_ih = Parameter(torch.Tensor(embedding_dim, hidden_dim*4))
        self.weight_hh = Parameter(torch.Tensor(hidden_dim, hidden_dim*4))
        self.bias = Parameter(torch.Tensor(hidden_dim*4))
        self.init_weights()
        self.linear = nn.Linear(self.hidden_dim, vocab_size)
        self.softmax = nn.LogSoftmax()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, input, hidden=None):
        length = input.size()[1]
        batch_size = input.size()[0]
        embeds = self.embeddings(input).view((batch_size, length, -1))
        # print(embeds)
        bs, seq_sz, _ = embeds.size()
        hidden_seq = []
        if hidden is None:
            h_t, c_t = (torch.zeros(self.hidden_dim).to(embeds.device), torch.zeros(self.hidden_dim).to(embeds.device))
        else:
            h_t, c_t = hidden

        for t in range(seq_sz):
            x_t = embeds[:, t, :]
            gates = x_t @ self.weight_ih + h_t @ self.weight_hh + self.bias
            i_t, f_t, g_t, o_t = gates.chunk(4, 1)
            i_t = torch.sigmoid(i_t)
            f_t = torch.sigmoid(f_t)
            g_t = torch.tanh(g_t)
            o_t = torch.sigmoid(o_t)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1)

        output = F.relu(self.linear(hidden_seq))
        if output.size()[1] == 1:  # seq_length=1, 会导致softmax出问题[1,1,V] 要先改成[1,V]
            output = output[:, 0, :]
            output = self.softmax(output)
            output = output.view(1, 1, -1)
            return output, (h_t, c_t)
        output = self.softmax(output)
        return output, (h_t, c_t)
