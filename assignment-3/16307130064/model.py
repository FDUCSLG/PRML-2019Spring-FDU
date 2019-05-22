import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
import torch.utils.data as Data
import torchvision.transforms as transforms
import math
class LSTMblock(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers=1):
        super(LSTMblock,self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.layers=num_layers
        self.wf=nn.ModuleList([nn.Linear(input_size+hidden_size,hidden_size) for i in range(num_layers)])
        self.wi=nn.ModuleList([nn.Linear(input_size+hidden_size,hidden_size) for i in range(num_layers)])
        self.wc=nn.ModuleList([nn.Linear(input_size+hidden_size,hidden_size) for i in range(num_layers)])
        self.wo=nn.ModuleList([nn.Linear(input_size+hidden_size,hidden_size) for i in range(num_layers)])
        self.init_weight()
    
    def init_weight(self):
        for p in self.parameters():
            torch.nn.init.uniform_(p,-1/math.sqrt(self.hidden_size),1/math.sqrt(self.hidden_size))

    def forward(self,inputs,state=None):
        batch_size,length, _ =inputs.size()
        outputs=torch.zeros(batch_size,length,self.hidden_size)
        if self.wf[0].weight.is_cuda:
            outputs=outputs.cuda()
        if state==None:
            state=self.initial_state(batch_size)
        
        for i in range(length):
            x=inputs[:,i,:]
            pht,pct=state
            ht,ct=self.initial_state(batch_size)
            for j in range(self.layers):
                z=torch.cat([pht[j],x],dim=-1)
                ft=torch.sigmoid(self.wf[j](z))
                it=torch.sigmoid(self.wi[j](z))
                ctt=torch.tanh(self.wc[j](z))
                ct[j]=ft*pct[j]+it*ctt
                ot=torch.sigmoid(self.wo[j](z))
                ht[j]=ot*torch.tanh(ct[j])             
                
            state=(ht,ct)
            outputs[:,i,:]=ht[-1]
            
        return outputs,state
        
    def initial_state(self,batch_size):
        h,c=torch.zeros(self.layers, batch_size,self.hidden_size),torch.zeros(self.layers, batch_size,self.hidden_size)
        if self.wf[0].weight.is_cuda:
            h,c=h.cuda(),c.cuda()            
        return (h,c)
        
class Generator(nn.Module):
    def __init__(self, embed, embed_dim, ntoken, hidden_size,num_layers=1, dropout=0.0, use_API=False):
        super(Generator,self).__init__()
        self.embed=embed
        if use_API==True:
            self.rnn=nn.LSTM(embed_dim,hidden_size,num_layers=num_layers,batch_first=True,bidirectional=False)       
        else: 
            print("write my own")
            self.rnn=LSTMblock(embed_dim, hidden_size,num_layers)
        self.drop=nn.Dropout(dropout)
        self.proj=nn.Linear(hidden_size, ntoken)
        
    def forward(self,inputs,pre_state=None):
        batch_size,length=inputs.size()
        x=self.drop(self.embed(inputs))
        out, state=self.rnn(x,pre_state)
        out=self.drop(out)
        out=self.proj(out)
        return out.view(batch_size*length,-1),state
