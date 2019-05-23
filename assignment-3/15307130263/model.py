import torch
from torch import nn
from torch import functional as F
import math
from random import choice

class HCLSTMCell(nn.Module):
    def __init__(self,input_size, hidden_size):
        super(HCLSTMCell, self).__init__()
        self.gatef = nn.Linear(input_size + hidden_size, hidden_size)
        self.gatei = nn.Linear(input_size + hidden_size, hidden_size)
        self.gatec = nn.Linear(input_size + hidden_size, hidden_size)
        self.gateo = nn.Linear(input_size + hidden_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        torch.nn.init.uniform_(self.gatef.weight,a=-1/math.sqrt(hidden_size),b=1/math.sqrt(hidden_size))
        torch.nn.init.uniform_(self.gatei.weight,a=-1/math.sqrt(hidden_size),b=1/math.sqrt(hidden_size))
        torch.nn.init.uniform_(self.gatec.weight,a=-1/math.sqrt(hidden_size),b=1/math.sqrt(hidden_size))
        torch.nn.init.uniform_(self.gateo.weight,a=-1/math.sqrt(hidden_size),b=1/math.sqrt(hidden_size))

    def forward(self, xt, ht_1, ct_1):
        z = torch.cat((ht_1,xt),1)
        ft = self.sigmoid(self.gatef(z))
        it = self.sigmoid(self.gatei(z))
        cbt = self.tanh(self.gatec(z))
        ct = torch.add(torch.mul(ct_1 , ft) , torch.mul(it , cbt))
        ot = self.gateo(z)
        ht = torch.mul(ot,self.tanh(ct))
        return ht,ct

class HCLSTM(nn.Module):
    def __init__(self,numEmbd,weight,hidden_size=128,usePreEmbedding = False,embedding = None):
        super(HCLSTM,self).__init__()
        self.numEmbd = numEmbd
        self.hiddenSize = hidden_size
        self.embdChannel = hidden_size
        self.weight = weight
        self.embedding = nn.Embedding(numEmbd,hidden_size)
        if usePreEmbedding:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding))
        self.cell = HCLSTMCell(
            input_size = hidden_size,
            hidden_size = hidden_size,
            )
        self.fc = nn.Linear(hidden_size,numEmbd)
        self.loss = nn.CrossEntropyLoss()#weight=weight)

        torch.nn.init.uniform_(self.fc.weight,a=-1/math.sqrt(hidden_size),b=1/math.sqrt(hidden_size))

    def forward(self,input_s,output_s):
        x = self.embedding(input_s)
        width = input_s.shape[1]
        bsize = input_s.shape[0]
        pred = []
        ht_1 = torch.zeros(input_s.shape[0],self.hiddenSize).cuda()
        ct_1 = torch.zeros(input_s.shape[0],self.hiddenSize).cuda()
        for t in range(width):
            it = x[:,t,:]
            ht_1,ct_1 = self.cell(it,ht_1,ct_1)
            ht = ht_1.view(bsize,1,self.hiddenSize)
            pred.append(ht)

        x = torch.cat(pred,1)
        x = self.fc(x)
        y = output_s
        x = x.view(bsize * width,-1)
        y = y.view(bsize * width)
        loss = self.loss(x,y)

        #maxd,maxP = torch.max(x,1)
        #acc = 0
        #for i in range(bsize * width):
        #    if maxP[i] == y[i]:
        #        acc += 1
        #print(acc / (bsize * width))

        x = x.view(bsize,width,-1)

        return {'pred' : x,'loss' : loss}
    
    def runStartWith(self,word,vocab,maxlength=24):
        eosid = vocab.to_index('<EOS>')
        idx = vocab.to_index(word)
        idx = torch.LongTensor([idx])
        x = self.embedding(idx)
        ht_1 = torch.zeros(1,self.hiddenSize)
        ct_1 = torch.zeros(1,self.hiddenSize)
        pred = []
        nowl = 0
        while 1:
            it = x
            ht_1,ct_1 = self.cell(it,ht_1,ct_1)
            x = ht_1
            x_t = self.fc(x)
            pred.append(x_t)
            maxd,maxP = torch.max(x_t,1)
            if maxP == eosid:
                break
            nowl += 1
            if nowl >= maxlength:
                break
        return pred
            

    def convertOutput(self,output,vocab,k=10):
        output = torch.cat(output,0)
        maxd,maxP = torch.topk(output,k,1)
        ret = []
        maxP = maxP.numpy().astype(int)
        for item in maxP:
            print(item)
            t = choice(item)
            ret.append(vocab.to_word(t))
        return ret