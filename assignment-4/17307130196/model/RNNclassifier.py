from fastNLP import CrossEntropyLoss
from math import sqrt
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

class MyLSTM(nn.Module):
    def __init__(self,D_input,D_H):
        super(MyLSTM,self).__init__()
        self.D_input=D_input
        self.D_H=D_H
        
        # Load moduled functions
        self.sigmoid=nn.Sigmoid()
        self.tanh=nn.Tanh()

        self.b=Parameter(torch.rand(4*self.D_H)*sqrt(2/(4*self.D_H)))
        self.W=Parameter(torch.rand(4*self.D_H,self.D_input)*sqrt(2/(4*self.D_H*self.D_H)))
        self.U=Parameter(torch.rand(4*self.D_H,self.D_H)*sqrt(2/(4*self.D_H*self.D_H)))

    def forward(self,inputs,c_last,h_last): # every input is a char
        gates = torch.mm(inputs,self.W.t())+torch.mm(h_last,self.U.t()) + self.b
        
        gi,gf,go,gc=gates.chunk(4,1)
        
        gi=self.sigmoid(gi)
        gf=self.sigmoid(gf)
        go=self.sigmoid(go)
        gc=self.tanh(gc)
        c=gf*c_last+gi*gc
        
        h=go*self.tanh(c)

        return h,c

class BiLSTM(nn.Module):
    def __init__(self,D_input,D_H):
        super(BiLSTM,self).__init__()
        self.D_input=D_input
        self.D_H=D_H
        self.LSTM_l=MyLSTM(D_input,D_H)
        self.LSTM_r=MyLSTM(D_input,D_H)

    def forward(self,embedded):
        embedded=embedded.permute(1,0,2)
        D_batch=embedded.shape[1]
        h_l=torch.zeros(D_batch,self.D_H)
        c_l=torch.zeros(D_batch,self.D_H)
        sl=embedded.shape[0]

        for i in range(sl):
            h_l,c_l=self.LSTM_l.forward(embedded[i],c_l,h_l)
        
        h_r=torch.zeros(embedded.shape[1],self.D_H)
        c_r=torch.zeros(embedded.shape[1],self.D_H)
        sl=embedded.shape[0]
        for i in reversed(range(sl)):
            h_r,c_r=self.LSTM_l.forward(embedded[i],c_r,h_r)

        hc=torch.cat((h_l,h_r),dim=1)
        return hc


class LSTMText(nn.Module):
    def __init__(self,D_input,D_H,D_output,D_dict=2001):
        super().__init__()
        self.D_dict=D_dict
        self.D_input=D_input
        self.D_H=D_H
        self.D_output=D_output
        self.Embedding=Parameter(torch.randn(D_dict,D_input))
        self.dropout=nn.Dropout(0.1)
        # Here is a bi-LSTM
        self.LSTM=BiLSTM(D_input,D_H)

        self.outputlayer=nn.Linear(2*D_H,D_output)

    def forward(self,words):
        D_batch=len(words)
        sl=len(words[0])
        embedded=torch.zeros(D_batch,sl,self.D_input)
        
        for sen_ct in range(D_batch):
            words_out=torch.zeros(sl,self.D_dict)
            for word_ct in range(sl):
                voc_id=words[sen_ct][word_ct]
                words_out[word_ct][voc_id]=1
            embedded[sen_ct]=torch.mm(words_out,self.Embedding)
                
        # embedded=self.dropout(self.embedding(words_out))
        hc = self.LSTM(embedded)
        hc = self.dropout(hc)
        pred = self.outputlayer(hc)
        # print(pred.shape)
        # pred = torch.softmax(pred,dim=0)
        return {'pred':pred}
    
    def predict(self,words):
        pred_dict=self.forward(words)
        pred=torch.zeros(len(words))
        pred_batch=torch.argmax(pred_dict['pred'],dim=1)
        for i in range(len(words)):
            pred[i]=pred_batch[i]
        return {'output':pred}

