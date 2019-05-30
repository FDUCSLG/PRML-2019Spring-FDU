import torch
import torch.nn as nn
from torch.nn import functional as F

class CNN(nn.Module):
    def __init__(self,vocab_size, embedding_size, class_num):
        super(CNN, self).__init__()
        kernel_num = 64
        kernel_size = (3, 4, 5)
        dropout = 0.5

        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.conv11 = nn.Conv2d(1, kernel_num, (kernel_size[0], embedding_size))
        self.conv12 = nn.Conv2d(1, kernel_num, (kernel_size[1], embedding_size))
        self.conv13 = nn.Conv2d(1, kernel_num, (kernel_size[2], embedding_size))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_size) * kernel_num, class_num)

    @staticmethod
    def conv_and_pool(x, conv):# x: (batch, 1, seq_len, embedding_size)
        x = conv(x) # x: (batch, kernel_num, _, 1)
        x = F.leaky_relu(x.squeeze(3))# x: (batch, kernel_num, _)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)#x:  (batch, kernel_num)
        return x

    def forward(self, input):
        x = self.embed(input)  
        x = x.unsqueeze(1)  #(batch, 1, seq_len, embedding_size)
        x1 = self.conv_and_pool(x, self.conv11)  # (batch, kernel_num)
        x2 = self.conv_and_pool(x, self.conv12)  # (batch, kernel_num)
        x3 = self.conv_and_pool(x, self.conv13)  # (batch, kernel_num)
        x = torch.cat((x1, x2, x3), 1)  # (batch, 3 * kernel_num)
        x = self.dropout(x)
        output = self.fc(x)
        return {"output":output}

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, class_num, rnn_type):
        super(RNN,self).__init__()
        self.rnn_type = self.rnn_type
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, class_num)

    def attention(self, lstm_output, final_state):
        merged_state = torch.cat([s for s in final_state], 1)#(batch,2*hidden)
        weights = torch.bmm(lstm_output, merged_state.unsqueeze(2)).squeeze(2)#(batch_size,seq_len)
        weights = F.softmax(weights,dim=1)
        atten_output =  torch.bmm(lstm_output.transpose(1, 2), weights.unsqueeze(2)).squeeze(2)#(batch,2*hidden)
        return atten_output

    def forward(self, input):
        x = input.transpose(1,0) #input：(batch,seq_len)->(seq_len,batch)
        embed = self.embed(x)
        x, (hidden, _)= self.lstm(embed) #hidden:(2,batch,hidden)
        x = x.transpose(1,0) #x:(seq_len,batch,2*hidden)->(batch,seq_len,2*hidden)
        if self.rnn_type == "max":
            x = torch.max(x,dim=1)[0]
        elif self.rnn_type == "min":
            x = torch.min(x,dim=1)[0]
        elif self.rnn_type == "mean":
            x = torch.mean(x,dim=1)
        elif self.rnn_type == "attention":
            x = self.attention(x, hidden)
        output = self.fc(x)
        return {"output":output}
       