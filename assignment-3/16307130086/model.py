import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np

class lstmPoem(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, cell_dim=128):
        super(lstmPoem, self).__init__()
        self.hidden_dim = hidden_dim
        self.cell_dim = cell_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        input_dim = hidden_dim + embedding_dim
        
        #init weight
        self.Wc = Parameter(torch.rand(cell_dim,input_dim) * np.sqrt(2/(input_dim + cell_dim)))
        self.Bc = Parameter(torch.rand(cell_dim, 1))
        self.Wf = Parameter(torch.rand(cell_dim,input_dim) * np.sqrt(2/(input_dim + cell_dim)))
        self.Bf = Parameter(torch.rand(cell_dim, 1))
        self.Wi = Parameter(torch.rand(cell_dim,input_dim) * np.sqrt(2/(input_dim + cell_dim)))
        self.Bi = Parameter(torch.rand(cell_dim, 1))
        self.Wo = Parameter(torch.rand(cell_dim,input_dim) * np.sqrt(2/(input_dim + cell_dim)))
        self.Bo = Parameter(torch.rand(cell_dim, 1))
        self.W = Parameter(torch.rand(vocab_size, hidden_dim) * np.sqrt(2/(vocab_size + hidden_dim)))
        self.b = Parameter(torch.rand(vocab_size, 1))


#         self.gate = nn.Linear(input_dim, cell_dim)
#         self.output = nn.Linear(hidden_dim, vocab_size)
#         self.sigmoid = nn.Sigmoid()
#         self.tanh = nn.Tanh()
    
    def forward(self, x, hidden=None, cell=None):
        
#         x:    seq_len * batch_size
        seq_len, batch_size = x.size()
    
        if hidden is None:
            Ht = x.data.new(self.hidden_dim, batch_size).fill_(0).float()
        else:
            Ht = hidden
            
        if cell is None:
            Ct = x.data.new(self.cell_dim, batch_size).fill_(0).float()
        else:
            Ct = cell
        
        embeds = self.embedding(x)
        # seq * batch * embedding
        
        output= []
        
        
        for i in range(len(embeds)):
            
            # self.Bx: cell_dim * 1
            # Wx:      cell_dim * input_dim
            # x_h:     input_dim * batch_size
            # C:       cell_dim * batch_size
            # H:       hidden_dim * batch_size
        
            xTmp = embeds[i].transpose(1,0).contiguous()
            x_h = torch.cat((xTmp, Ht),0).cuda()
            
            Ft = torch.sigmoid(self.Wf.mm(x_h) + self.Bf)
            It = torch.sigmoid(self.Wi.mm(x_h) + self.Bi)
            Ot = torch.sigmoid(self.Wo.mm(x_h) + self.Bo)
            Ct_ = torch.tanh(self.Wc.mm(x_h) + self.Bc)
            
            Ct = torch.add(torch.mul(Ft, Ct), torch.mul(It, Ct_))
            Ht = torch.mul(torch.tanh(Ct), Ot)
            y = self.W.mm(Ht) + self.b
            # no softmax: included in cross entropy loss
            y = y.transpose(1,0).contiguous()
            # y:  batch_size, vocab
            output.append(y)
            
        output = torch.cat(output,0)
        output = output.view(seq_len * batch_size, -1)
        #output:    (seq * batchsize, vocab)
        return output, Ht, Ct

    
class GRUPoem(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128):
        super(GRUPoem, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        input_dim = hidden_dim + embedding_dim
        
        #init weight
        self.Wr = Parameter(torch.rand(hidden_dim,input_dim) * np.sqrt(2/(input_dim + hidden_dim)))
        self.Br = Parameter(torch.rand(hidden_dim, 1))
        self.Wz = Parameter(torch.rand(hidden_dim,input_dim) * np.sqrt(2/(hidden_dim + hidden_dim)))
        self.Bz = Parameter(torch.rand(hidden_dim, 1))
        self.Wh = Parameter(torch.rand(hidden_dim,input_dim) * np.sqrt(2/(input_dim + hidden_dim)))
        self.Bh = Parameter(torch.rand(hidden_dim, 1))
        self.W = Parameter(torch.rand(vocab_size, hidden_dim) * np.sqrt(2/(vocab_size + hidden_dim)))
        self.b = Parameter(torch.rand(vocab_size, 1))

    def forward(self, x, hidden=None):
#         x:    seq_len * batch_size
        seq_len, batch_size = x.size()
    
        if hidden is None:
            Ht = x.data.new(self.hidden_dim, batch_size).fill_(0).float()
        else:
            Ht = hidden
            
        
        embeds = self.embedding(x)
        # seq * batch * embedding
        
        output= []
        
        
        for i in range(len(embeds)):
            
            xTmp = embeds[i].transpose(1,0).contiguous()
            x_h = torch.cat((xTmp, Ht),0).cuda()
                
            Rt = torch.sigmoid(self.Wr.mm(x_h) + self.Br)
            Zt = torch.sigmoid(self.Wz.mm(x_h) + self.Bz)
            Ht_ = torch.mul(Ht, Rt)
            x_h_ = torch.cat((xTmp, Ht_),0).cuda()
            H_ = torch.tanh(self.Wh.mm(x_h_) + self.Bh)
            
            Ht = torch.mul(Zt, Ht)+torch.mul((1 - Zt, H_))
            y = self.W.mm(Ht) + self.b
            # no softmax: included in cross entropy loss
            y = y.transpose(1,0).contiguous()
            # y:  batch_size, vocab
            output.append(y)
            
        output = torch.cat(output,0)
        output = output.view(seq_len * batch_size, -1)
        #output:    (seq * batchsize, vocab)
        return output, Ht
