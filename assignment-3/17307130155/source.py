import json 
from langconv import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import os
import sys
from fastNLP import DataSet,Instance,Vocabulary,Instance
import numpy as np
sys.path.append('..')

vocab = Vocabulary(min_freq=5)
data_dim = 256
hidden_dim = 128
PATH = 'tang1.pth'
max_length = 48
batch_size = 20

def split_sent(ins):
        answer = []
        for a in ins['raw_sentence']:
                answer.append(a);
        return answer

def get_data():
        global max_length
        dataset = DataSet()
        for i in range(58):
                f = open("../../../chinese-poetry/json/poet.tang."+str(i*1000)+".json", encoding='utf-8')  
                setting = json.load(f)
                #print(setting)
                for line in setting:
                        if (len(line['paragraphs'])==4 and len(line['paragraphs'][0])==12 and len(line['paragraphs'][1])==12 and len(line['paragraphs'][2])==12 and len(line['paragraphs'][3])==12):
                                s=''
                                for sentence in line['paragraphs']:
                                        s+=Converter('zh-hans').convert(sentence)
                                s+='$'
                                dataset.append(Instance(raw_sentence=s))
                f.close()
                print('Has processed '+str((i+1)*1000)+' poems')
        
        dataset.apply(split_sent, new_field_name='words')
        train_data,test_data = dataset.split(0.2)
        return train_data,test_data

def build_vocabulary(train_data,test_data):
        train_data.apply(lambda x: [vocab.add(word) for word in x['words']])
        vocab.build_vocab()
        train_data.apply(lambda x: [vocab.to_index(word) for word in x['words']], new_field_name='words')
        test_data.apply(lambda x: [vocab.to_index(word) for word in x['words']], new_field_name='words')
        return train_data,test_data

class mylstm(nn.Module):
        def __init__(self,data_dim,hidden_dim,onehot_dim,batchsize):    
                super(mylstm, self).__init__()
                self.data_dim = data_dim
                self.hidden_dim = hidden_dim
                self.batchsize = batchsize
                self.onehot_dim = onehot_dim
                self.embedding = nn.Embedding(self.onehot_dim,self.data_dim)

                self.output = nn.Linear(self.hidden_dim,self.onehot_dim,bias=True)

                self.Wf = nn.Parameter(torch.rand(self.hidden_dim,self.data_dim), requires_grad=True)
                self.Wi = nn.Parameter(torch.rand(self.hidden_dim,self.data_dim), requires_grad=True)
                self.Wc = nn.Parameter(torch.rand(self.hidden_dim,self.data_dim), requires_grad=True)
                self.Wo = nn.Parameter(torch.rand(self.hidden_dim,self.data_dim), requires_grad=True)
                
                self.Uf = nn.Parameter(torch.rand(self.hidden_dim,self.hidden_dim), requires_grad=True)
                self.Ui = nn.Parameter(torch.rand(self.hidden_dim,self.hidden_dim), requires_grad=True)
                self.Uc = nn.Parameter(torch.rand(self.hidden_dim,self.hidden_dim), requires_grad=True)
                self.Uo = nn.Parameter(torch.rand(self.hidden_dim,self.hidden_dim), requires_grad=True)

                self.bf = nn.Parameter(torch.rand(self.hidden_dim,1), requires_grad=True)
                self.bi = nn.Parameter(torch.rand(self.hidden_dim,1), requires_grad=True)
                self.bc = nn.Parameter(torch.rand(self.hidden_dim,1), requires_grad=True)
                self.bo = nn.Parameter(torch.rand(self.hidden_dim,1), requires_grad=True)
                self.sigmoid = nn.Sigmoid()
                self.tanh = nn.Tanh()

        def forward(self,x_raw,h_last,c_last):

                x = self.embedding(x_raw).t()

                f = self.sigmoid(self.Wf.mm(x)+self.Uf.mm(h_last)+self.bf)
                i = self.sigmoid(self.Wi.mm(x)+self.Ui.mm(h_last)+self.bi)
                c_hat = self.tanh(self.Wc.mm(x)+self.Uc.mm(h_last)+self.bc)
                c = torch.mul(f,c_last)+torch.mul(i,c_hat)
                o = self.sigmoid(self.Wo.mm(x)+self.Uo.mm(h_last)+self.bo)
                z = torch.mul(o,self.tanh(c))

                h = self.output(z.t())
                return z,c,h


def generate(x):
        s=''
        h_last = torch.zeros(hidden_dim,1)
        #print(h_last)
        c_last = torch.zeros(hidden_dim,1)
        i = 0
        while (x!=vocab['$']):
                s+=vocab.to_word(x)
                #print(x)
                input = torch.LongTensor([x])
                h,c,predict = lstm.forward(input,h_last,c_last)
                h_last = h
                c_last = c
                t = torch.argmax(predict).item()
                #print(t)
                x = t
                i+= 1
                if (i==120):
                        break
        print(s)
        
                


dataset,test_data = get_data()
print('Data loaded. There are '+str(len(dataset))+ ' poems in dataset.')
#print(dataset[0])
dataset,test_data = build_vocabulary(dataset,test_data)
print('Vocabulary has been built.')
print('There are '+str(len(vocab))+' words in dictionary.')

lstm = mylstm(data_dim,hidden_dim,len(vocab),batch_size)
#lstm.load_state_dict(torch.load(PATH))
epoch = 0
last_perpecity = 10000
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(lstm.parameters())
#optimizer = optim.Adam(lstm.parameters(),lr=1e-2)
#optimizer = optim.SGD(lstm.parameters(),lr=1e-2)
#optimizer = optim.Adagrad(lstm.parameters(),lr=1e-2)
#optimizer = optim.Adadelta(lstm.parameters(),lr=1e-2)
record = []
while (epoch<40):
        epoch += 1
        loss_t = 0
        for index in range(len(dataset)//batch_size):
                h_last = torch.zeros(hidden_dim,1)
                c_last = torch.zeros(hidden_dim,1)
                l = torch.FloatTensor([0])
                for k in range(0,max_length):
                        lstm.zero_grad()
                   
                        x_raw = Variable(torch.LongTensor([dataset[j]['words'][k] for j in range(index,index+batch_size)]))
                       
                        h,c,predict = lstm(x_raw,h_last,c_last)
                        t = Variable(torch.LongTensor([dataset[j]['words'][k+1] for j in range(index,index+batch_size)]))
                       
                        loss = criterion(predict,t)
                        c_last = c
                        h_last = h
                        l = l + loss

                l = l/(max_length)
                l.backward()
                loss_t += l.item()
                optimizer.step()

                if ((index+1)%10 ==0):
                        print('Epoch '+str(epoch)+' has train '+str((index+1)*batch_size)+' poems, the loss is :'+str(loss_t/(index+1)))
        
        with torch.no_grad():
                loss_in_test = 0
                for index in range(len(test_data)):
                        h_last = torch.zeros(hidden_dim,1)
                        c_last = torch.zeros(hidden_dim,1)
                        l = torch.FloatTensor([0])
                        for k in range(0,max_length):
                                lstm.zero_grad()
                                x_raw = Variable(torch.LongTensor([test_data[j]['words'][k] for j in range(index,index+1)]))
                        
                                h,c,predict = lstm(x_raw,h_last,c_last)
                                t = Variable(torch.LongTensor([test_data[j]['words'][k+1] for j in range(index,index+1)]))
                        
                                loss = criterion(predict,t)
                                c_last = c
                                h_last = h
                                loss_in_test = loss_in_test + loss.item()/max_length
                                criterion.zero_grad()
                
        perplexity = np.exp(loss_in_test/(len(test_data)))
        #print(loss_in_test/(len(test_data)))
        print('Average perpecity of epoch '+str(epoch)+' : '+str(perplexity))
        

'''
lstm = mylstm(data_dim,hidden_dim,len(vocab),batch_size)
lstm.load_state_dict(torch.load(PATH))
'''
print('The record of perplexity :')
print(record)
generate(vocab['月'])
generate(vocab['日'])
generate(vocab['红'])
generate(vocab['山'])
generate(vocab['夜'])
generate(vocab['湖'])
generate(vocab['海'])
#generate(vocab['黄'])
#generate(vocab['诗'])
#generate(vocab['萦'])
    

           