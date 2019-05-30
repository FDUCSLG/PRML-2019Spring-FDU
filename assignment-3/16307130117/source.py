import json
import torch
import numpy as np
import os
import torch.autograd as autograd
import torch.nn as nn
from torch import nn
import torch.optim as optim 
import re
import math
import string
from langconv import *
import sys

##由于代码是在jupyter notebook上书写的，所以整理成python文件比较混乱
##其实如果可以直接交ipynb文件会方便很多，可以直接看结果和注释
##如果助教要运行检查的话可以一块块贴于notebook上，当然运行前需要先设置好数据集，将全唐诗数据集直接放于当前目录下的json文件夹内，保留原命名



###################读取数据集######################
'''
filedir = os.getcwd()+'/json'
filenames=os.listdir(filedir)
filelist = []
data = []
for i in filenames:
    if i[:9] == "poet.tang":
        filelist.append(i)

for filename in filelist:
    filepath = filedir + '/' + filename
    with open(filepath, 'rb') as f:
        dat = json.load(f)
    data.append(dat)
'''

###################处理数据集########################
'''
def cleantxt(raw):
    fil = re.compile(r'[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+')
    return re.sub(fil,'', raw) 

def cht_to_chs(line):
    line = Converter('zh-hans').convert(line)
    line.encode('utf-8')
    return line

s_len = 60
poems = []
data_w = data[:2]
for i in data_w:
    for j in i:
        temp = ""
        for k in j['paragraphs']:
            temp+=k
        
        temp = cleantxt(temp)
        if(len(temp)<20):
            continue
        if(len(temp) < s_len):
            temp = ' '*(s_len-len(temp)) + temp
        poems.append(cht_to_chs(temp[:s_len]))  ##转简体  
print(poems[0])
print(len(poems))
len_train = math.ceil(len(poems)*0.8)

train_data_w = poems[:len_train]
#train_data_w = train_data_w[:10000]
dev_data_w = poems[len_train+1:]

from fastNLP import Vocabulary
from fastNLP import DataSet
from fastNLP import Instance

vocab = Vocabulary()

for sentence in train_data_w:
    for word in sentence:
        vocab.add(word)
vocab.add('?')
vocab.build_vocab() ##字典


train_data = []
dev_data = []
for sentence in train_data_w:
    sent = []
    for word in sentence:
        sent.append(vocab.to_index(word))
    train_data.append(sent)

for sentence in dev_data_w:
    sent = []
    for word in sentence:
        if(word in vocab):
            sent.append(vocab.to_index(word))
        else:
            sent.append(vocab.to_index('?'))
    dev_data.append(sent) ##转向量

'''


###################Pytorch version########################
'''
class poetrymodel(nn.Module):
    def __init__(self,input_size, hidden_size, target_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.target_size = target_size
        self.embedding = nn.Embedding(target_size,input_size)
        self.w_f = nn.Parameter(torch.rand(hidden_size, input_size + hidden_size)*np.sqrt(2/(hidden_size*2+input_size)))
        self.w_i = nn.Parameter(torch.rand(hidden_size, input_size + hidden_size)*np.sqrt(2/(hidden_size*2+input_size)))
        self.w_c = nn.Parameter(torch.rand(hidden_size, input_size + hidden_size)*np.sqrt(2/(hidden_size*2+input_size)))
        self.w_o = nn.Parameter(torch.rand(hidden_size, input_size + hidden_size)*np.sqrt(2/(hidden_size*2+input_size)))
        self.b_f = nn.Parameter(torch.rand(hidden_size, 1))
        self.b_i = nn.Parameter(torch.rand(hidden_size, 1))
        self.b_c = nn.Parameter(torch.rand(hidden_size, 1))
        self.b_t = nn.Parameter(torch.rand(hidden_size, 1))
        self.w = nn.Parameter(torch.rand(target_size, hidden_size)*np.sqrt(2/(hidden_size+target_size)))
        self.b = nn.Parameter(torch.rand(target_size, 1))
        #self.hh = autograd.Variable(torch.rand(self.hidden_size, batch_size))
        #self.cc = autograd.Variable(torch.rand(self.hidden_size, batch_size))
        

    
    def forward(self, inp, batch_size, init_states=None):
        if init_states == None:
            h = torch.zeros(self.hidden_size,batch_size).cuda()
            c = torch.zeros(self.hidden_size,batch_size).cuda()
        else:
            h, c = init_states
        T = inp.shape[0]

        target = []
        #z = z.type(torch.FloatTensor)
        for t in range(T):
            temp = self.embedding(inp[t]).permute(1,0)
            z = torch.cat((temp,h),0)
            f_t = self.gate(self.w_f,z,self.b_f,"sigmoid")   #torch.sigmoid(self.w_f.mm(z[t]) + self.b_f)
            i_t = self.gate(self.w_i,z,self.b_i,"sigmoid")   #torch.sigmoid(self.w_i.mm(z[t]) + self.b_i)
            g_t = self.gate(self.w_c,z,self.b_c,"tanh")   #torch.tanh(self.w_c.mm(z[t])+self.b_c)
            c = f_t *c + i_t*g_t
            o_t = self.gate(self.w_o,z,self.b_t,"sigmoid")#torch.sigmoid(self.w_o.mm(z) + self.b_t)
            h = o_t * torch.tanh(c)
            a = self.w.mm(h)+self.b
            target.append(torch.log_softmax(a,0))
            #target.append(a)
        target = torch.cat(target, dim = 0).reshape(T,self.target_size,batch_size)
        return target,(h,c)
    
    def test(self,x):
        return self.w
    
    def gate(self,w,z,b,char):
        if char == "sigmoid":
            return torch.sigmoid(w.mm(z)+b)
        else:
            return torch.tanh(w.mm(z)+b)
    
    def predict(self, x):
        target = self.forward(x)
        pre_y = np.argmax(target.reshape(len(x), -1).numpy(), axis=1)         
        return pre_y
    
    def loss(self,x, y):
        cost = 0        
        for i in range(len(y)):
            target = self.forward(x[i])
            pre_yi = target[range(len(y[i])), y[i]]
            cost -= np.sum(np.log(pre_yi))

        N = np.sum([len(yi) for yi in y])
        ave_loss = cost / N

        return ave_loss
    
    def generate(self, x,state):
        out, (h,c) = self.forward(x,state)
        pred = out[-1]
        pred_y = np.argmax(pred.reshape(len(x), -1).numpy(), axis=1)         
        return pred_y,(h,c)

batch_size = 32
train_d = []
dev_d = []
for d in range(math.floor(len(train_data)/batch_size)):
    inp = torch.tensor([np.zeros((len(train_data[d])-1))]*batch_size) 
    y = torch.zeros(batch_size,len(train_data[d])-1)
    for j in range(batch_size):
        data = train_data[d*batch_size+j]
        inp[j] = torch.LongTensor(data[:-1])
        data = torch.from_numpy(np.asarray(data))    
        y[j] = data[1:]
    train_d.append((inp.permute(1,0),y.permute(1,0)))

for data in dev_data:
    inp = torch.tensor([np.zeros((len(data)-1))]*1) 
    inp[0] = torch.LongTensor(data[:-1])
    y = torch.zeros(1,len(train_data[d])-1)
    data = torch.from_numpy(np.asarray(data))    
    y[0] = data[1:]
    dev_d.append((inp.permute(1,0),y.permute(1,0)))

model = poetrymodel(64,128,len(vocab)).cuda()
#model = LSTMTagger(10,16,1,len(vocab))
loss_function = nn.CrossEntropyLoss() 
optimizer = optim.Adam(model.parameters(), lr=1e-3)
model= torch.load("C:/Users/C-changing/Desktop/PR/model2.pth")


def perplexity(model,s_len):
    pp_sum = 0
    s = 1
    for data,y in dev_d:
        s = 1
        score,_ = model(data.long().cuda(),1)
        #score = torch.softmax(score,0)
        for j in range(len(score)):
            temp = score[j][int(y[j][0])][0]
            s *= math.exp(temp)
            temp = pow(1/s, 1/len(score))
        pp_sum += temp
    return (pp_sum/len(dev_d))/s_len

epochs = 100
batch = 1
losses = []
cc = 2019
for e in range(epochs):
    
    if(e > 0 and e % 5 == 0):
        temp = perplexity(model,s_len)
        if temp > cc:
            break
        else:
            print("old_perplexity:{}".format(cc))
            cc = temp
            print("perplexity:{}".format(cc))
    train_loss = 0
    rank = 0
    for data,y in train_d:
        rank += 1
        score,state = model(data.long().cuda(),batch_size)
       # print(state)
        loss = loss_function(score, y.long().cuda())
        optimizer.zero_grad()
        loss.backward()
        #print(loss.item())
        #nn.utils.clip_grad_norm(model.parameters(), 5)
        optimizer.step()
        train_loss += loss.item()
        if(rank % 100 == 0):
            print(loss.item())
    print("epoch:{} , train_loss = {}".format(e,train_loss/rank))
    losses.append(train_loss/rank)
    train_loss = 0
    rank = 0
'''



############################Pytorch 生成#####################
'''
def generate(model,x,state):
    out, (h,c) = model(x,1,state)
    pred = out.data[-1].reshape(model.target_size)
    #pred[vocab.to_index(' ')] = 0
    print(pred[86])
    pred_y,pred_label = torch.topk(pred, 10, 0)#选几个
    pred_y /= torch.sum(pred_y)
    pred_y = pred_y.squeeze(0).cpu().numpy()
    pred_label = pred_label.squeeze(0).cpu().numpy()
    result = np.random.choice(pred_label, size=1, p=pred_y)

    return result,(h,c)

state = None
result = ''
w = '月'
result+= w
for i in range(27):
    w_in= torch.tensor([[vocab.to_index(w)]]).long()
    word, state = generate(model,w_in.cuda(),state)
    w = vocab.to_word(int(word[0]))
    result+=w
print(result)
c = vocab.to_word(40)
#torch.save(model,"C:/Users/C-changing/Desktop/PR/model2.pth")
'''

################Numpy version#######################
'''
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def tanh(x):
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

def logsoftmax(x):
    m = max(x)
    exp_x = np.exp(x-m)
    Z = np.sum(exp_x)
    return x-m-np.log(Z)

def softmax(x):
    if x.shape[1] == 1:
        temp = max(x)
        x -= temp
        return (np.exp(x)/np.sum(np.exp(x)))
    temp = np.max(x)
    x -= np.ones(x.shape)*temp
    vec = np.dot(np.exp(x.T),np.ones((x.shape[0],1)))
    vec = 1/(vec*np.ones((1,x.shape[0])))
    result = np.multiply(np.exp(np.mat(x)),vec.T)
    return result

class poetrymodel2():
    def __init__(self,input_size, hidden_size, target_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.target_size = target_size
        self.embedding = nn.Embedding(target_size,input_size)
        self.w_f = np.random.random((hidden_size, input_size + hidden_size))*np.sqrt(2/(hidden_size*2+input_size))
        self.w_i = np.random.random((hidden_size, input_size + hidden_size))*np.sqrt(2/(hidden_size*2+input_size))
        self.w_c = np.random.random((hidden_size, input_size + hidden_size))*np.sqrt(2/(hidden_size*2+input_size))
        self.w_o = np.random.random((hidden_size, input_size + hidden_size))*np.sqrt(2/(hidden_size*2+input_size))
        self.b_f = np.random.random((hidden_size, 1))
        self.b_i = np.random.random((hidden_size, 1))
        self.b_c = np.random.random((hidden_size, 1))
        self.b_t = np.random.random((hidden_size, 1))
        self.w = np.random.random((target_size, hidden_size))*np.sqrt(2/(hidden_size+target_size))
        self.b = np.random.random((target_size, 1))
        #self.hh = autograd.Variable(torch.rand(self.hidden_size, batch_size))
        #self.cc = autograd.Variable(torch.rand(self.hidden_size, batch_size))
        
    def init_m(self,dim,batch,T):
        return np.zeros((T+1,dim, batch))
    
    def forward(self, inp, batch_size, init_states=None):
        T = inp.shape[0]
        h = np.zeros((T+1,self.hidden_size,batch_size))
        c = np.zeros((T+1,self.hidden_size,batch_size))
        if init_states != None:
            h[-1], c[-1] = init_states
        
        f = self.init_m(self.hidden_size,batch_size,T)
        i = self.init_m(self.hidden_size,batch_size,T)
        g = self.init_m(self.hidden_size,batch_size,T)
        o = self.init_m(self.hidden_size,batch_size,T)
        z = self.init_m(self.hidden_size+self.input_size,batch_size,T)
        target = []
        #z = z.type(torch.FloatTensor)
        for t in range(T):

            temp = self.embedding(torch.tensor(inp[t]).long()).permute(1,0).detach().numpy()
            z[t] = np.concatenate((temp,h[t-1]),0)
            f[t] = self.gate(self.w_f,z[t],self.b_f,"sigmoid")   #torch.sigmoid(self.w_f.mm(z[t]) + self.b_f)
            i[t] = self.gate(self.w_i,z[t],self.b_i,"sigmoid")   #torch.sigmoid(self.w_i.mm(z[t]) + self.b_i)
            g[t] = self.gate(self.w_c,z[t],self.b_c,"tanh")   #torch.tanh(self.w_c.mm(z[t])+self.b_c)
            c[t] = f[t] *c[t-1] + i[t]*g[t]
            o[t] = self.gate(self.w_o,z[t],self.b_t,"sigmoid")#torch.sigmoid(self.w_o.mm(z) + self.b_t)
            h[t] = o[t] * tanh(c[t])
            a = self.w.dot(h[t])+self.b
            target.append(softmax(a))
        target = np.array(target).reshape(T,self.target_size,batch_size)
        return target,(z,f,i,g,o,h,c)
    
    def gate(self,w,z,b,char):
        if char == "sigmoid":
            return sigmoid(w.dot(z)+b)
        else:
            return tanh(w.dot(z)+b)
    
    def predict(self, x):
        target = self.forward(x)
        pre_y = np.argmax(target.reshape(len(x), -1).numpy(), axis=1)         
        return pre_y
    
    def test(self):
        return self.w
    
    def loss(self,score, y):
        loss = 0        
        for j in range(score.shape[2]):
            cc = np.array([score[i][int(y[i][j])][j] for i in range(len(y))])
            pre_yi = np.log(cc)
            loss -= np.sum(pre_yi)
        ave_loss = (loss / len(y))/y.shape[1]
        return ave_loss
    
    def init_grad(self,batch):
        dw = np.zeros(self.w_i.shape)
        db = np.zeros((self.b_i.shape[0],batch))
        return dw,db
    
    def bp(self, target, y,state,step):
        #print(target)
        dw_f, db_f = self.init_grad(target.shape[-1])
        dw_i, db_i = self.init_grad(target.shape[-1])
        dw_o, db_o = self.init_grad(target.shape[-1])
        dw_c, db_c = self.init_grad(target.shape[-1])
        dw,db = np.zeros((self.w.shape[0],self.w.shape[1])),np.zeros((self.b.shape[0],target.shape[-1]))
        dct = np.zeros((self.hidden_size,target.shape[-1]))
        
        (z,f,i,g,o,h,c) = state
        dt = target
        #print(y)
        for t in range(len(y)):
            for j in range(len(y[t])):
                dt[t][int(y[t][j])][j] -= 1
        
        for tt in range(len(y)):
            t = int(len(y) - tt - 1)
            zz = z[t].transpose((1,0))
            dw += dt[t].dot(h[t].transpose((1,0)))
            db += dt[t]

            dht = self.w.T.dot(dt[t])
            
            dot = dht * tanh(c[t])
            dct += dht * o[t] * (1-tanh(c[t])**2)
            dct1 = dct * f[t]
            dit = dct * g[t]
            dft = dct * c[t-1]
            dgt = dct * i[t]
            
            dg = dgt * (1-g[t]**2)
            di = dit * i[t] * (1-i[t])
            df = dft * f[t] * (1-f[t])
            do = dot * o[t] * (1-o[t])
            
            dw_c += dg.dot(zz)
            db_c += dg
            dw_i += di.dot(zz)
            db_i += di
            dw_f += df.dot(zz)
            db_f += df
            dw_o += do.dot(zz)
            db_o += do
            dct = dct1
            dct1 = np.zeros((self.hidden_size,target.shape[-1]))
        
        
        step = step/target.shape[-1]
        self.w_f -= step * dw_f
        temp = np.sum(db_f,axis = 1)
        self.b_f -= step * temp.reshape((len(temp),1))
        self.w_i -= step * dw_i
        temp = np.sum(db_i,axis = 1)
        self.b_i -= step * temp.reshape((len(temp),1))
        self.w_o -= step * dw_o
        temp = np.sum(db_o,axis = 1)
        self.b_t -= step * temp.reshape((len(temp),1))
        self.w_c -= step * dw_c
        temp = np.sum(db_c,axis = 1)
        self.b_c -= step * temp.reshape((len(temp),1))
        self.w -= step * dw
        temp = np.sum(db,axis = 1)
        self.b -= step * temp.reshape((len(temp),1))

batch_size = 32
train_d = []
for d in range(math.floor(len(train_data)/batch_size)):
    inp = np.array([np.zeros((len(train_data[d])))]*batch_size) 
    y = np.zeros((batch_size,len(train_data[d])))
    for j in range(batch_size):
        data = train_data[d*batch_size+j]
        inp[j] = np.asarray(data)
        data = np.asarray(data)    
        y[j][:-1],y[j][-1] = data[1:],data[0]
    train_d.append((inp.transpose(1,0),y.transpose(1,0)))
model2 = poetrymodel2(64,128,len(vocab))

epochs = 20
batch = batch_size
step = 0.001
losses = []
for e in range(epochs):
    train_loss = 0
    rank = 0
    t_loss = 0
    for data,y in train_d:
        rank += 1
        score,state = model2.forward(data,batch_size)
        loss = model2.loss(score,y)
        train_loss += loss
        t_loss += loss
        model2.bp(score,y,state,step)
        if(rank % 100 == 0):

            print(t_loss/100)
            t_loss = 0

    print("epoch:{} , train_loss = {}".format(e,train_loss/rank))
    losses.append(train_loss/rank)
    train_loss = 0
    rank = 0

def generate_2(model,x,state):
    out, (z,f,i,g,o,h,c) = model.forward(x,1,state)
    pred = out[-1].reshape(model.target_size)
    pred[vocab.to_index(' ')] = 0
    pred_y,pred_label = torch.topk(torch.tensor(pred), 30, 0)#选几个
    pred_y /= torch.sum(pred_y)
    pred_y = pred_y.squeeze(0).cpu().numpy()
    pred_label = pred_label.squeeze(0).cpu().numpy()
    result = np.random.choice(pred_label, size=1, p=pred_y)

    return result,(h[-2],c[-2])

state = None
result = []
w = '日'
for i in range(19):
    w_in= np.asarray([[vocab.to_index(w)]])
    word, state = generate_2(model2,w_in,state)
    w = vocab.to_word(int(word[0]))
    result.append(w)
print(result)
c = vocab.to_word(40)
'''