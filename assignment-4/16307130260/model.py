import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from fastNLP import Const

class MyCNNText(nn.Module):
    def __init__(self, class_num, vocab_size, embed_dim=128, kernel_num=100,
                kernel_sizes=(3,4,5), dropout=0.5, embed_weights=None):
        super(MyCNNText, self).__init__()
        
        V = vocab_size
        D = embed_dim
        C = class_num
        Ci = 1
        Co = kernel_num
        Ks = kernel_sizes

        self.dropout = dropout

        self.embed = nn.Embedding(V, D)
        if embed_weights != None:
            # print(embed_weights)
            embed_weights = np.array(embed_weights).astype(np.float32)
            self.embed.weight.data.copy_(torch.from_numpy(embed_weights))

        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(Ks)*Co, C)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, words):
        x = self.embed(words)  # (N, W, D)

        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)

        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)

        return {Const.OUTPUT: logit}

    def predict(self, words):
        pred = None
        with torch.no_grad():
            x = self.embed(words)  # (N, W, D)
            # if self.static:
            #     x = Variable(x)
            x = x.unsqueeze(1)  # (N, Ci, W, D)
            x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
            x = torch.cat(x, 1)
            logit = self.fc1(x)  # (N, C)
            pred = logit.max(1)
        return {"pred": pred.indices}


class MyLSTMText(nn.Module):
    def __init__(self, class_num, vocab_size, embed_dim=128, hidden_size=256, dropout=1, embed_weights=None):
        super(MyLSTMText, self).__init__()
        self.V = vocab_size
        self.D = embed_dim
        self.H = hidden_size
        self.C = class_num

        self.dropout = dropout

        self.embed = nn.Embedding(self.V, self.D)

        if embed_weights != None:
            # print(embed_weights)
            embed_weights = np.array(embed_weights).astype(np.float32)
            self.embed.weight.data.copy_(torch.from_numpy(embed_weights))

        self.lstm = nn.LSTM(self.D, self.H, dropout=dropout)
        self.fc = nn.Linear(self.H, self.C)
    
    def forward(self, words):
        # W, N = words.size()
        # h_0 = torch.zeros(1, N, self.H).float()
        # c_0 = torch.zeros(1, N, self.H).float()

        # words size: (N, W)
        words = words.transpose(1, 0)

        # (W, N, D)
        x = self.embed(words)


        # _, (hidden_h, _) = self.lstm(x, (h_0, c_0))
        output, (hidden_h, _) = self.lstm(x)
        output_pooling = F.max_pool1d(output.permute(1, 2, 0), output.size(0)).squeeze(2)
        # print(output_pooling.size())

        logit = self.fc(output_pooling)
        # logit = self.fc(hidden_h[-1])

        return {Const.OUTPUT: logit}
    
    def predict(self, words):
        pred = None
        with torch.no_grad():
            output = self(words)
            pred = output[Const.OUTPUT].max(1)
        return {"pred": pred.indices}

class MyBLSTMText(nn.Module):
    def __init__(self, class_num, vocab_size, embed_dim=128, hidden_size=256, dropout=1, embed_weights=None):
        super(MyBLSTMText, self).__init__()
        self.V = vocab_size
        self.D = embed_dim
        self.H = hidden_size
        self.C = class_num

        self.dropout = dropout

        self.embed = nn.Embedding(self.V, self.D)
        
        if embed_weights != None:
            # print(embed_weights)
            embed_weights = np.array(embed_weights).astype(np.float32)
            self.embed.weight.data.copy_(torch.from_numpy(embed_weights))

        self.lstm = nn.LSTM(self.D, self.H, bidirectional=True,dropout=dropout)
        self.fc = nn.Linear(2 * self.H, self.C)
    
    def forward(self, words):
        # W, N = words.size()
        # h_0 = torch.zeros(1, N, self.H).float()
        # c_0 = torch.zeros(1, N, self.H).float()

        # words size: (N, W)
        words = words.transpose(1, 0)

        # (W, N, D)
        x = self.embed(words)


        # _, (hidden_h, _) = self.lstm(x, (h_0, c_0))
        output, (hidden_h, _) = self.lstm(x)
        output_pooling = F.max_pool1d(output.permute(1, 2, 0), output.size(0)).squeeze(2)
        # print(output_pooling.size())

        logit = self.fc(output_pooling)
        # logit = self.fc(hidden_h[-1])

        return {Const.OUTPUT: logit}
    
    def predict(self, words):
        pred = None
        with torch.no_grad():
            output = self(words)
            pred = output[Const.OUTPUT].max(1)
        return {"pred": pred.indices}