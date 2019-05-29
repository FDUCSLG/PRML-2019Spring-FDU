import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from fastNLP.modules import encoder
class Embedding(nn.Module):
    def __init__(self, char_size, embed_dim, init_embedding=None):
        super(Embedding, self).__init__()
        if init_embedding is None:
            self.lut=nn.Embedding(char_size,embed_dim)  
            self.embed_dim = embed_dim
        else:
            freeze=False
            self.lut=nn.Embedding.from_pretrained(torch.FloatTensor(init_embedding), freeze=freeze)
            _ , self.embed_dim = init_embedding.shape    
            print("freeze:",freeze)
        
    def forward(self, x):
        return self.lut(x)
                
class RNNText(nn.Module): 
    def __init__(self, src_embed, hidden_size,tag_size,dropout=0.2,layers=2):
        super(RNNText, self).__init__()
        self.lstm = nn.LSTM(input_size = src_embed.embed_dim,
                            hidden_size = hidden_size,
                            num_layers = layers,
                            dropout = dropout,
                            batch_first=True,
                            bidirectional = True)
        self.src_embed = src_embed
        self.proj = nn.Linear(hidden_size*2*layers, tag_size)
        self.tag_size=tag_size
        self.drop=nn.Dropout(dropout)

    def forward(self, words, seq_len=None):
        batch_size,_=words.size()
        out, (h,c) =self.lstm(self.src_embed(words))
        feat=self.drop(h.permute(1,0,2).contiguous().view(batch_size,-1))
        out = self.proj(feat)
        return {"pred":out}
        
    def predict(self, words, seq_len=None):
        output = self(words, seq_len)
        _, predict = output["pred"].max(dim=1)
        return {"pred":predict}
        
class CNNText(torch.nn.Module):   
    def __init__(self, src_embed, num_classes, kernel_nums=(25, 25, 25),
                 kernel_sizes=(3, 4, 5),
                 padding=0,
                 dropout=0.5):
        super(CNNText, self).__init__()        
        self.src_embed = src_embed
        self.conv_pool = encoder.ConvMaxpool(
            in_channels=src_embed.embed_dim,
            out_channels=kernel_nums,
            kernel_sizes=kernel_sizes,
            padding=padding)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(sum(kernel_nums), num_classes)
    
    def forward(self, words, seq_len=None):
        x = self.src_embed(words)  # [N,L] -> [N,L,C]
        x = self.conv_pool(x)  # [N,L,C] -> [N,C]
        x = self.dropout(x)
        x = self.fc(x)  # [N,C] -> [N, N_class]
        return {"pred":x}
    
    def predict(self, words, seq_len=None):
        output = self(words, seq_len)
        _, predict = output["pred"].max(dim=1)
        return {"pred":predict}

class CNN_KMAX(nn.Module):
    def __init__(self, src_embed, num_classes, kernel_nums=(50, 50, 50),
                 kernel_sizes=(3, 4, 5),
                 padding=0,
                 dropout=0.5,k=2):
        super(CNN_KMAX, self).__init__()
        self.src_embed = src_embed
        self.fc = nn.Linear(sum(kernel_nums)*k, num_classes)
        self.k = k
        self.convs = nn.ModuleList([nn.Conv2d(1, C, (K, src_embed.embed_dim),padding=padding) for K,C in zip(kernel_sizes,kernel_nums)])
        self.dropout = nn.Dropout(dropout)

    def kmax_pool(self, x):
        index = x.topk(self.k, dim = 2)[1].sort(dim = 2)[0]
        return x.gather(2, index)
        
    def forward(self, words, seq_len=None):
        x = self.src_embed(words) 
        #print(x.size())
        bs=x.size(0)
        x = x.unsqueeze(1) 
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs] 
        x = [self.kmax_pool(i).squeeze(2) for i in x] 
        x = torch.cat(x, 1).contiguous().view(bs, -1)
        x = self.dropout(x) 
        #print(x.size())
        x = self.fc(x)  
        return {"pred":x}
