import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import os
import sys
from fastNLP import DataSet,Instance,Vocabulary,Instance
import numpy as np
sys.path.append('..')

vocab = Vocabulary(min_freq=1)
data_dim = 64
hidden_dim = 64
PATH = 'parameters.pth'

def split_sent(ins):
        answer = []
        for a in ins['raw_sentence']:
                answer.append(a);
        return answer

def add_end(ins):
        return ins['raw_sentence']+'$'

def get_data():
        s = ''
        dataset = DataSet()
        for line in open('../handout/tangshi.txt'):
                if (line == '\n'):
                        dataset.append(Instance(raw_sentence=s, label='0'))
                        #print(s)
                        s = ''
                else :
                        s += line.replace('\n','')

        dataset.apply(add_end, new_field_name='raw_sentence')
        dataset.apply(split_sent, new_field_name='words')
        return dataset

def preprocess(ins):
        ans = np.zeros((len(ins['words']),len(vocab)))
        for a in range(len(ins['words'])):
                ans[a][ins['words'][a]] = 1
        return ans

def build_vocabulary(train_data):
        train_data.apply(lambda x: [vocab.add(word) for word in x['words']])
        vocab.build_vocab()
        train_data.apply(lambda x: [vocab.to_index(word) for word in x['words']], new_field_name='words')
        return train_data

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
                c = torch.mul(f,c_last.reshape(-1,1))+torch.mul(i,c_hat)
                o = self.sigmoid(self.Wo.mm(x)+self.Uo.mm(h_last)+self.bo)
                z = torch.mul(o,self.tanh(c))

                h = self.output(z.t())
                return z,c,h


def generate(x):
        s=''
        h_last = torch.zeros(hidden_dim,1)
        #print(h_last)
        c_last = torch.zeros(hidden_dim,1)
        while (x!=vocab['$']):
                s+=vocab.to_word(x)
                input = torch.LongTensor([x])
                h,c,predict = lstm.forward(input,h_last,c_last)
                h_last = h
                c_last = c
                t = torch.argmax(predict).item()
                #print(t)
                x = t
        print(s)
        
                


dataset = get_data()
#print(dataset[1])
dataset = build_vocabulary(dataset)
print(len(vocab))

lstm = mylstm(data_dim,hidden_dim,len(vocab),1)
#lstm.load_state_dict(torch.load(PATH))
epoch = 70
print(dataset)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(lstm.parameters(),lr=1e-2)

for i in range(epoch):
        loss_t = 0
        for j in dataset[1:100]:
                h_last = torch.zeros(hidden_dim,1)
                c_last = torch.zeros(hidden_dim,1)
                l = torch.FloatTensor([0])
                for k in range(0,(len(j['words'])-1)):
                        lstm.zero_grad()
                        x_raw = Variable(torch.LongTensor([j['words'][k]]))
                        
                        h,c,predict = lstm(x_raw,h_last,c_last)
                        t = Variable(torch.LongTensor([j['words'][k+1]]))

                        #print(vocab.to_word(j['words'][k])+' '+vocab.to_word(j['words'][k+1]))
                        loss = criterion(predict,t)
                        c_last = c
                        h_last = h
                        l = l + loss

                if (len(j['words']) != 1):
                        l = l/(len(j['words'])-1)
                        l.backward()
                        loss_t += l.item()
                        optimizer.step()
        if ((i+1)%10 == 0):
                torch.save(lstm.state_dict(), PATH)
        print('Perpecxity of epoch '+str(i+1)+': '+str(loss_t/len(dataset)))



'''
lstm = mylstm(data_dim,hidden_dim,len(vocab),1)
lstm.load_state_dict(torch.load(PATH))
'''
#generate(vocab['月'])
#generate(vocab['日'])
#generate(vocab['红'])
#generate(vocab['山'])
#generate(vocab['夜'])
#generate(vocab['湖'])
#generate(vocab['海'])
#generate(vocab['黄'])
#generate(vocab['诗'])
#generate(vocab['萦'])
    
