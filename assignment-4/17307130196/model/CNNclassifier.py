from fastNLP import CrossEntropyLoss
from torch.nn import Parameter
import torch.nn.functional as F
import torch
import torch.nn as nn
from math import sqrt

class CNNcond(nn.Module):
    def __init__(self,D_input,D_ker):
        # D-vec is the size of embedding
        # D-ker is the size of kernel
        super(CNNcond,self).__init__()
        self.D_input=D_input
        self.D_ker=D_ker
        self.filter=Parameter(torch.randn(D_input*D_ker,1)*sqrt(2/(self.D_ker*self.D_input)))
        self.bias=Parameter(torch.randn(1))
    
    def one(self,seq_word):
        # input will be ( D_ker, D_batch , D_input )
        # output will be (D_batch)
        # (D_batch, D_ker, D_input)
        seq_word=seq_word.permute(1,0,2)
        D_batch=len(seq_word)
        seq_word=seq_word.reshape(D_batch,-1,1).squeeze(dim=2)
        return torch.mm(seq_word,self.filter).squeeze(dim=1)+self.bias

    def forward(self,embedded):
        # input: a batch of sl
        # (D_batch, sl, D_input)
        # output will be a batch of cond
        # (D_batch, sl)
        D_batch=len(embedded)
        sl=len(embedded[0])
        result=torch.rand(sl,D_batch)
        padding=torch.zeros(D_batch,self.D_ker-1,self.D_input)
        # embedded: (D_batch, sl+ker-1,D_input)
        embedded=torch.cat((embedded,padding),dim=1).permute(1,0,2)
        
        for i in range(sl):
            In=embedded[i:i+self.D_ker]
            result[i]=self.one(In)

        return result.t()


class CNNText(nn.Module):
    def __init__(self,D_input,D_output,upper=5,lower=3,D_dict=2001,layernum=5):
        super(CNNText,self).__init__()
        self.D_dict=D_dict
        self.D_input=D_input
        self.D_output=D_output
        self.upper=upper
        self.lower=lower
        self.D_cond=(upper-lower)*layernum
        self.layernum=layernum
        self.Embedding=Parameter(torch.rand(D_dict,D_input)*sqrt(2/(D_dict*D_input)))
        self.dropout=nn.Dropout(0.1)

        self.Cond=[]
        for i in range(lower,upper):
            for k in range(layernum):
                self.Cond.append(CNNcond(D_input,i))
        self.relu=nn.ReLU()
        self.outputlayer=Parameter(torch.randn(self.D_cond,D_output))

        print('Model has been built!')

    def forward(self,words):
        # input will be a batch of sentences
        # (D_batch, sl)
        D_batch=len(words)
        sl=len(words[0])
        embedded=torch.zeros(D_batch,sl,self.D_input)
        
        # embedded is after embedding layer
        # (D_batch, sl, D_input)
        for sen_ct in range(D_batch):
            words_out=torch.zeros(sl,self.D_dict)
            for word_ct in range(sl):
                voc_id=words[sen_ct][word_ct]
                words_out[word_ct][voc_id]=1
            embedded[sen_ct]=torch.mm(words_out,self.Embedding)

        # Then data should be pushed into Cond layers
        # output of the layer: (D_batch, sl)
        # (D_cond,D_batch,sl)
        cond_res=torch.rand(self.D_cond,D_batch,sl)
        for i in range(self.D_cond):
            cond_res[i]=self.Cond[i](embedded)
        cond_res=self.relu(cond_res)
        # (D_batch,D_cond,sl)
        cond_res=cond_res.permute(1,0,2)
        
        # (D_batch,D_cond)
        max_pool=cond_res.max(dim=2).values
        # print(max_pool)
        pred=torch.mm(max_pool,self.outputlayer)
        # pred=pred-pred.max()

        return {'pred':pred}
        # Then the max pooling layers
        
    def predict(self,words):
        pred_dict=self.forward(words)
        pred=torch.zeros(len(words))
        pred_batch=torch.argmax(pred_dict['pred'],dim=1)
        for i in range(len(words)):
            pred[i]=pred_batch[i]
        return {'output':pred}

