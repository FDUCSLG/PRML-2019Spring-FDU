import json
from math import sqrt
# import random
import torch

from data.preposess import *

print('Processing')
dict=build_pretrain_dict()
dictionary,dictionary_rev=build_dict()
D_dict=len(dictionary)
eight_lines,eight_lines_dev=clr_poem(dictionary,dictionary_rev)
num_of_poems=len(eight_lines)
num_of_poems_dev=len(eight_lines_dev)
print(num_of_poems)
print("The Dictionary has been built!")
#===================================================================
# Parameters
D_batch=32
D_input=300
D_H=128
T=40

char_each_poem=len(eight_lines[0])


def sigmoid(x): 
    return 1. / (1 + torch.exp(-x))

class LSTMstate:
    def __init__(self,D_H,D_input):
        self.i=torch.zeros(D_H)
        self.f=torch.zeros(D_H)
        self.o=torch.zeros(D_H)
        self.c=torch.zeros(D_H)
        self.h=torch.zeros(D_H)
        self.c_last=torch.zeros(D_H)
        self.h_last=torch.zeros(D_H)
        
        self.h_grad=torch.zeros(D_H)
        self.c_grad=torch.zeros(D_H)
        self.x=torch.zeros(D_input)
    
    def forward(self,x,para,h_last,c_last): # every input is a char    
        self.h_last=h_last
        self.c_last=c_last
        self.x=x

        self.i=torch.sigmoid(torch.mm(x,para.Wi.t())+torch.mm(h_last,para.Ui.t()) + para.bi)
        self.f=torch.sigmoid(torch.mm(x,para.Wf.t())+torch.mm(h_last,para.Uf.t()) + para.bf)
        self.o=torch.sigmoid(torch.mm(x,para.Wo.t())+torch.mm(h_last,para.Uo.t()) + para.bo)
        self.h=torch.tanh(torch.mm(x,para.Wh.t())+torch.mm(h_last,para.Uh.t()) + para.bh)
        
        # print(self.c.shape)
        # print(c_last.shape)
        self.c=self.f*c_last+self.i*self.h
        
        self.h=self.o*torch.tanh(self.c)
        return self.h,self.c

class LSTMpara:
    def __init__(self,D_input,D_H):
        self.D_input=D_input
        self.D_H=D_H

        self.bo=torch.rand(D_H)*sqrt(2/(D_H))
        self.bh=torch.rand(D_H)*sqrt(2/(D_H))
        self.bi=torch.rand(D_H)*sqrt(2/(D_H))
        self.bf=torch.rand(D_H)*sqrt(2/(D_H))

        self.bo_grad=torch.zeros(D_H)
        self.bh_grad=torch.zeros(D_H)
        self.bi_grad=torch.zeros(D_H)
        self.bf_grad=torch.zeros(D_H)
        
        self.Wo=torch.rand(D_H,D_input)*sqrt(2/(D_H*D_H))
        self.Uo=torch.rand(D_H,D_H)*sqrt(2/(D_H*D_H))
        self.Wo_grad=torch.zeros(D_H,D_input)
        self.Uo_grad=torch.zeros(D_H,D_H)

        self.Wh=torch.rand(D_H,D_input)*sqrt(2/(D_H*D_H))
        self.Uh=torch.rand(D_H,D_H)*sqrt(2/(D_H*D_H))
        self.Wh_grad=torch.zeros(D_H,D_input)
        self.Uh_grad=torch.zeros(D_H,D_H)

        self.Wi=torch.rand(D_H,D_input)*sqrt(2/(D_H*D_H))
        self.Ui=torch.rand(D_H,D_H)*sqrt(2/(D_H*D_H))
        self.Wi_grad=torch.zeros(D_H,D_input)
        self.Ui_grad=torch.zeros(D_H,D_H)

        self.Wf=torch.rand(D_H,D_input)*sqrt(2/(D_H*D_H))
        self.Uf=torch.rand(D_H,D_H)*sqrt(2/(D_H*D_H))
        self.Wf_grad=torch.zeros(D_H,D_input)
        self.Uf_grad=torch.zeros(D_H,D_H)
    
    def update(self,lr):

        self.Wo-=lr*self.Wo_grad
        self.Wf-=lr*self.Wf_grad
        self.Wi-=lr*self.Wi_grad
        self.Wh-=lr*self.Wh_grad

        self.Uo-=lr*self.Uo_grad
        self.Uf-=lr*self.Uf_grad
        self.Ui-=lr*self.Ui_grad
        self.Uh-=lr*self.Uh_grad

        self.bo-=lr*self.bo_grad
        self.bf-=lr*self.bf_grad
        self.bi-=lr*self.bi_grad
        self.bh-=lr*self.bh_grad
        print('Grad')
        print(self.Wo_grad.sum())

        # print(self.Wo_grad.sum())
        # print(self.Wf_grad.sum())
        # print(self.Wi_grad.sum())
        # print(self.Wh_grad.sum())

        # print(self.Uo_grad.sum())
        # print(self.Uf_grad.sum())
        # print(self.Ui_grad.sum())
        # print(self.Uh_grad.sum())

        self.Wo_grad=0
        self.Wf_grad=0
        self.Wi_grad=0
        self.Wh_grad=0

        self.Uo_grad=0
        self.Uf_grad=0
        self.Ui_grad=0
        self.Uh_grad=0

        self.bo_grad=0
        self.bf_grad=0
        self.bi_grad=0
        self.bh_grad=0

        # input('whatever')

    # store the grad without update
    def grad(self,h_grad,c_grad,state):
        # print(h_grad)
        d_c=state.o*h_grad*(1-state.c*state.c)+c_grad
        d_h=d_c*state.i
        d_f=d_c*c_grad
        d_o=h_grad*torch.tanh(state.c)
        d_i=d_c*state.h

        d_h_=d_h*(1-state.h*state.h)
        d_f_=d_f*state.f*(1-state.f)
        d_i_=d_i*state.i*(1-state.i)
        d_o_=d_o*state.o*(1-state.o)

        # print(c_grad)
        self.Wi_grad+=torch.mm(d_i_.t(),state.x)
        self.Wo_grad+=torch.mm(d_o_.t(),state.x)
        self.Wf_grad+=torch.mm(d_f_.t(),state.x)
        self.Wh_grad+=torch.mm(d_h_.t(),state.x)
        
        self.Ui_grad+=torch.mm(d_i_.t(),state.c)
        self.Uo_grad+=torch.mm(d_o_.t(),state.c)
        self.Uf_grad+=torch.mm(d_f_.t(),state.c)
        self.Uh_grad+=torch.mm(d_h_.t(),state.c)

        self.bi_grad+=d_i_.squeeze(dim=0)
        self.bo_grad+=d_o_.squeeze(dim=0)
        self.bf_grad+=d_f_.squeeze(dim=0)
        self.bh_grad+=d_h_.squeeze(dim=0)
        
        h_grad=torch.mm(d_i_,self.Ui)+torch.mm(d_f_,self.Uf)+torch.mm(d_h_,self.Uh)+torch.mm(d_o_,self.Uo)
        c_grad=d_c*state.f

        return h_grad,c_grad
        
class CorssEntropy:
    def loss(self, pred, targ):
        pred=torch.exp(pred)/torch.exp(pred).sum()
        loss=targ*torch.log(pred+0.00001)
        return -loss.sum()

    def grad(self, pred, targ):
        pred = torch.exp(pred)/torch.exp(pred).sum()
        # print('grad before: %f'%pred.sum())
        grad = targ/(pred+0.00001)
        # print('grad after %f'%grad.sum())
        return -grad/(2000)


def get_targ_vec(poems,char_ct):    
        D_batch=poems.shape[0]
        char_each_poem=poems.shape[1]

        char_tars=torch.rand(D_batch,self.D_dict)
        for poem_ct in range(D_batch):
            char_tars[poem_ct]=poems[poem_ct][char_ct+1]
        
        return char_tars

def one_hots(poem,dictionary):
    # Input is a list of poems. In fact, this is the batch of poems
    # Output should be a 3D matrics
    char_each_poem=len(poem)
    In=torch.zeros(char_each_poem,D_dict) 
    Out=torch.zeros(char_each_poem,D_dict) 
    # print(output.shape)    

    for char_ct in range(char_each_poem-1):
        char=poem[char_ct]
        In[char_ct][dictionary[char]]=1

    for char_ct in range(char_each_poem-1):
        char=poem[char_ct+1]     
        Out[char_ct][dictionary[char]]=1    
    
    return In,Out


class LSTMchain:
    def __init__(self,D_H,D_input,sl):
        self.D_H=D_H
        self.D_input=D_input
        self.sl=sl

        # store the state
        self.state_chain=[]
        # real parameter of this LSTM
        self.para=LSTMpara(D_input,D_H)
    
    # This is the function that get all the grad after forward is donw
    def get_grad(self,targs):
        h_prev=torch.zeros(1,self.D_H)
        c_prev=torch.zeros(1,self.D_H)
        loss=0
        for i in reversed(range(self.sl)):
            loss+=CorssEntropy().loss(self.state_chain[i].h,targs[i])
            # print(loss)
            h_prev_this=CorssEntropy().grad(self.state_chain[i].h,targs[i])
            # print('...')
            h_prev_this+=h_prev
            # print('Prev')
            # print(h_prev_this)
            # print(h_prev)
            h_prev,c_prev=self.para.grad(h_prev_this,c_prev,self.state_chain[i])
            # print('...')

        return loss
    
    # This is the function to get the chain
    def get_state(self,inputs):
        h_last=torch.zeros(1,self.D_H)
        c_last=torch.zeros(1,self.D_H)

        for i in range(self.sl):
            node=LSTMstate(self.D_input,self.D_H)

            # get the h_last and c_last
            h_last,c_last=node.forward(inputs[i].unsqueeze(dim=0),self.para,h_last,c_last)

            # add the state to chain
            self.state_chain.append(node)

    def clear_chain(self):
        self.state_chain=[]

    def train(self,inputs,targs):
        self.clear_chain()
        self.get_state(inputs)
        loss=self.get_grad(targs)
        self.para.update(0.00 1)
        return loss

def train_numpy():
    LSTM=LSTMchain(D_dict,D_input,char_each_poem)
    num_of_poems=len(eight_lines)    
    pretrain=torch.rand(D_dict,D_input)
    ct=0
    for char in dictionary.keys():
        if char in dict.keys():
            pretrain[ct]=torch.tensor(dict[char])
            ct+=1
    
    for t in range(T): # for each poem, train turns times
        Sum=0
        print("Epoch: T=%d" % (t))
        for i in range(num_of_poems): 
            inpu,targ=one_hots(eight_lines[i],dictionary)
            embedding=torch.mm(inpu,pretrain)
            loss=LSTM.train(embedding,targ)
            print(loss/char_each_poem)            
            Sum+=loss

train_numpy()