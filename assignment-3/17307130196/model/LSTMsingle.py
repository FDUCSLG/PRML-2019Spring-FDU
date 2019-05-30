import json
from math import sqrt
import random
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

import sys
sys.path.append("..")
from data.preposess import *
#==========================================================
# Models
# This is a version that update each char
class MyLSTM(nn.Module):
    def __init__(self,D_input,D_H):
        super(MyLSTM,self).__init__()
        self.D_input=D_input
        self.D_H=D_H
        
        # Load moduled functions
        self.sigmoid=nn.Sigmoid()
        self.tanh=nn.Tanh()
        b=torch.rand(4*self.D_H)*sqrt(2/(4*self.D_H))
        # for i in range(self.D_H,2*self.D_H):
        #     b[i]=1
        # bi,bf,bo,bc
        self.b=Parameter(b)
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

        # print(h)

        return c,h

class PoemProducerSingle(nn.Module):
    def __init__(self,D_input,D_H,D_dict,dictionary,dict):
        super(PoemProducerSingle,self).__init__()
        self.D_input=D_input
        self.D_dict=D_dict
        self.D_H=D_H
        pretrain=torch.rand(D_dict,D_input)
        # the embedding layer, turn the word with one-hot representation to 
        ct=0
        hit=0
        for char in dictionary.keys():
            if char in dict.keys():
                hit+=1
                pretrain[ct]=torch.tensor(dict[char])
            ct+=1
        print("hit %d chars!"%hit)
        
        # self.WordEmbedding=pretrain
        self.WordEmbedding=Parameter(pretrain)
        #LSTM with the input a D_input size embedding, output is a D_output vec
        self.LSTM=MyLSTM(D_input,D_H)
        
        #the FFNN layer, turn the output of LSTM and remap into the dictionary
        self.FFNN=Parameter(torch.rand(D_H,D_dict))
        print("Model has been built!")

    def forward(self,poems,char_ct,h_last,c_last):
        D_batch=poems.shape[0]
        char_each_poem=poems.shape[1]
        # Forward
        one_hots = torch.rand(D_batch,self.D_dict)
        for poem_ct in range(D_batch):
            one_hots[poem_ct]=poems[poem_ct][char_ct]
        
        x=torch.mm(one_hots,self.WordEmbedding)

        torch.nn.Dropout(0.35)            

        c , h = self.LSTM(x,c_last,h_last)
        y_pred_raw=torch.mm(h,self.FFNN)
        # y_pred_raw = self.FFNN(h)
        y_pred = F.softmax(y_pred_raw,dim=1)
        # print(y_pred)

        return y_pred,h,c

    def get_targ_vec(self,poems,char_ct):    
        D_batch=poems.shape[0]
        char_each_poem=poems.shape[1]

        char_tars=torch.rand(D_batch,self.D_dict)
        for poem_ct in range(D_batch):
            char_tars[poem_ct]=poems[poem_ct][char_ct+1]
        
        return char_tars

    def turn_char_poems_to_one_hots(self,poems,dictionary):
        # Input is a list of poems. In fact, this is the batch of poems
        # Output should be a 3D matrics
        poem_num=len(poems)
        char_each_poem=len(poems[0])
        output=torch.zeros(poem_num,char_each_poem,self.D_dict) 

        # print(output.shape)
        
        for poem_ct in range(poem_num):
            for char_ct in range(char_each_poem):
                char=poems[poem_ct][char_ct]
                output[poem_ct][char_ct][dictionary[char]]=1    

        return output
    
    def produce(self,char_begin,char_each_poem,dictionary,dictionary_rev):
        poem=[]

        h_last=torch.zeros(1,self.D_H)
        c_last=torch.zeros(1,self.D_H) 
        char='0'
        for char_ct in range(char_each_poem-1): 
            one_hot=torch.zeros(1,self.D_dict)
            if char_ct==0:
                char='*'
            elif char_ct==1:
                char=char_begin
            poem.append(char) 
            one_hot[0][dictionary[char]]=1        
            x=torch.mm(one_hot,self.WordEmbedding)
            c , h = self.LSTM.forward(x,c_last,h_last)
            
            y_pred_raw = torch.mm(h,self.FFNN)
            y_pred = F.softmax(y_pred_raw.squeeze())
            id_pred=torch.argmax(y_pred).item()
            
            h_last=h
            c_last=c
            char=dictionary_rev[id_pred]
        poem.append(char)
        return poem

class PoemProducerDouble(nn.Module):
    def __init__(self,D_input,D_H,D_dict,dictionary,dict):
        super(PoemProducerDouble,self).__init__()
        self.D_input=D_input
        self.D_dict=D_dict
        self.D_H=D_H

        pretrain=torch.rand(D_dict,D_input)
        #the embedding layer, turn the word with one-hot representation to 
        ct=0
        hit=0
        for char in dictionary.keys():
            if char in dict.keys():
                hit+=1
                pretrain[ct]=torch.tensor(dict[char])
            ct+=1
        
        print("hit %d chars!"%hit)
        
        self.WordEmbedding=Parameter(pretrain)
        #LSTM with the input a D_input size embedding, output is a D_output vec
        self.LSTM1=MyLSTM(D_input,D_H)
        self.LSTM2=MyLSTM(D_H,D_H)
        
        #the FFNN layer, turn the output of LSTM and remap into the dictionary
        self.FFNN=Parameter(torch.rand(D_H,D_dict))
        print("Model has been built!")

    def forward(self,poems,char_ct,h_last1,c_last1,h_last2,c_last2):
        D_batch=poems.shape[0]
        char_each_poem=poems.shape[1]
        # Forward
        one_hots = torch.rand(D_batch,self.D_dict)
        for poem_ct in range(D_batch):
            one_hots[poem_ct]=poems[poem_ct][char_ct]
        
        x=torch.mm(one_hots,self.WordEmbedding)
            
        torch.nn.Dropout(0.35)            

        c1 , h1 = self.LSTM1(x ,c_last1,h_last1)
        
        torch.nn.Dropout(0.35)           
        
        c2 , h2 = self.LSTM2(h1,c_last2,h_last2)
        y_pred_raw=torch.mm(h2,self.FFNN)
        # y_pred_raw = self.FFNN(h)
        y_pred = F.softmax(y_pred_raw,dim=1)
        # print(y_pred)

        return y_pred,h1,c1,h2,c2

    def get_targ_vec(self,poems,char_ct):    
        D_batch=poems.shape[0]
        char_each_poem=poems.shape[1]

        char_tars=torch.rand(D_batch,self.D_dict)
        for poem_ct in range(D_batch):
            char_tars[poem_ct]=poems[poem_ct][char_ct+1]
        
        return char_tars

    def turn_char_poems_to_one_hots(self,poems,dictionary):
        # Input is a list of poems. In fact, this is the batch of poems
        # Output should be a 3D matrics
        poem_num=len(poems)
        char_each_poem=len(poems[0])
        output=torch.zeros(poem_num,char_each_poem,self.D_dict) 

        # print(output.shape)
        
        for poem_ct in range(poem_num):
            for char_ct in range(char_each_poem):
                char=poems[poem_ct][char_ct]
                output[poem_ct][char_ct][dictionary[char]]=1    

        return output
    
    def produce(self,char_begin,char_each_poem,dictionary,dictionary_rev):
        poem=[]

        h_last1=torch.zeros(1,self.D_H)
        c_last1=torch.zeros(1,self.D_H)
        h_last2=torch.zeros(1,self.D_H)
        c_last2=torch.zeros(1,self.D_H) 
        char='0'
        for char_ct in range(char_each_poem-1): 
            one_hot=torch.zeros(1,self.D_dict)
            if char_ct==0:
                char='*'
            elif char_ct==1:
                char=char_begin
            poem.append(char) 
            one_hot[0][dictionary[char]]=1        
            x=torch.mm(one_hot,self.WordEmbedding)

            c1 , h1 = self.LSTM1.forward(x,c_last1,h_last1)
            c2 , h2 = self.LSTM2.forward(h1,c_last2,h_last2)
            

            y_pred_raw = torch.mm(h2,self.FFNN)
            y_pred = F.softmax(y_pred_raw.squeeze())
            id_pred=torch.argmax(y_pred).item()
            
            h_last1=h1
            c_last1=c1
            h_last2=h2
            c_last2=c2
            char=dictionary_rev[id_pred]
        poem.append(char)
        return poem

class PoemProducerTriple(nn.Module):
    def __init__(self,D_input,D_H,D_dict,dictionary,dict):
        super(PoemProducerTriple,self).__init__()
        self.D_input=D_input
        self.D_dict=D_dict
        self.D_H=D_H

        pretrain=torch.rand(D_dict,D_input)
        #the embedding layer, turn the word with one-hot representation to 
        ct=0
        hit=0
        for char in dictionary.keys():
            if char in dict.keys():
                hit+=1
                pretrain[ct]=torch.tensor(dict[char])
            ct+=1
        
        print("hit %d chars!"%hit)
        
        self.WordEmbedding=Parameter(pretrain)
        #LSTM with the input a D_input size embedding, output is a D_output vec
        self.LSTM1=MyLSTM(D_input,D_H)
        self.LSTM2=MyLSTM(D_H,D_H)
        self.LSTM3=MyLSTM(D_H,D_H)
        
        #the FFNN layer, turn the output of LSTM and remap into the dictionary
        self.FFNN=Parameter(torch.rand(D_H,D_dict))
        print("Model has been built!")

    def forward(self,poems,char_ct,h_last1,c_last1,h_last2,c_last2,h_last3,c_last3):
        D_batch=poems.shape[0]
        char_each_poem=poems.shape[1]
        # Forward
        one_hots = torch.rand(D_batch,self.D_dict)
        for poem_ct in range(D_batch):
            one_hots[poem_ct]=poems[poem_ct][char_ct]
        
        x=torch.mm(one_hots,self.WordEmbedding)
            
        torch.nn.Dropout(0.20)            

        c1 , h1 = self.LSTM1(x ,c_last1,h_last1)
        
        torch.nn.Dropout(0.20)            
        
        c2 , h2 = self.LSTM2(h1,c_last2,h_last2)

        torch.nn.Dropout(0.20)

        c3 , h3 = self.LSTM3(h2,c_last3,h_last3)

        y_pred_raw=torch.mm(h3,self.FFNN)
        # y_pred_raw = self.FFNN(h)
        y_pred = F.softmax(y_pred_raw,dim=1)
        # print(y_pred)

        return y_pred,h1,c1,h2,c2,h3,c3

    def get_targ_vec(self,poems,char_ct):    
        D_batch=poems.shape[0]
        char_each_poem=poems.shape[1]

        char_tars=torch.rand(D_batch,self.D_dict)
        for poem_ct in range(D_batch):
            char_tars[poem_ct]=poems[poem_ct][char_ct+1]
        
        return char_tars

    def turn_char_poems_to_one_hots(self,poems,dictionary):
        # Input is a list of poems. In fact, this is the batch of poems
        # Output should be a 3D matrics
        poem_num=len(poems)
        char_each_poem=len(poems[0])
        output=torch.zeros(poem_num,char_each_poem,self.D_dict) 

        # print(output.shape)
        
        for poem_ct in range(poem_num):
            for char_ct in range(char_each_poem):
                char=poems[poem_ct][char_ct]
                output[poem_ct][char_ct][dictionary[char]]=1    

        return output
    
    def produce(self,char_begin,char_each_poem,dictionary,dictionary_rev):
        poem=[]

        h_last1=torch.zeros(1,self.D_H)
        c_last1=torch.zeros(1,self.D_H)
        h_last2=torch.zeros(1,self.D_H)
        c_last2=torch.zeros(1,self.D_H)
        h_last3=torch.zeros(1,self.D_H)
        c_last3=torch.zeros(1,self.D_H) 
        char='0'
        for char_ct in range(char_each_poem-1): 
            one_hot=torch.zeros(1,self.D_dict)
            if char_ct==0:
                char='*'
            elif char_ct==1:
                char=char_begin
            poem.append(char) 
            one_hot[0][dictionary[char]]=1        
            x=torch.mm(one_hot,self.WordEmbedding)

            c1 , h1 = self.LSTM1.forward(x,c_last1,h_last1)
            c2 , h2 = self.LSTM2.forward(h1,c_last2,h_last2)
            c3 , h3 = self.LSTM3.forward(h2,c_last3,h_last3)
        
            y_pred_raw = torch.mm(h3,self.FFNN)
            y_pred = F.softmax(y_pred_raw.squeeze())
            id_pred=torch.argmax(y_pred).item()
            
            h_last1=h1
            c_last1=c1
            h_last2=h2
            c_last2=c2
            h_last3=h3
            c_last3=c3
            char=dictionary_rev[id_pred]
        poem.append(char)
        return poem

#===================================================================
# Assist functs

class CrossEntropy(nn.Module):
    def __init__(self):
        super(CrossEntropy, self).__init__()
    def forward(self,y_pred,y_tar,D_dict):

        log_y_pred=y_tar*torch.log(y_pred+0.000001)
        return -torch.sum(log_y_pred)


def produce(char,poemproducer,char_each_poem,dictionary,dictionary_rev):
    poem=poemproducer.produce(char,char_each_poem,dictionary,dictionary_rev)
    poem=reform(poem)

def cross_entropy(y_pred,y_tar,D_dict):
    log_y_pred=y_tar*torch.log(y_pred)
    return -torch.sum(log_y_pred,dim=1)
    
