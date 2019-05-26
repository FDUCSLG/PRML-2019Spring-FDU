from prepareData import *
from model import *
import random
import torch
import numpy as np
import torch.nn as nn

#utils

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def tanh(x):
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
    
def preprocess(path='./json', batch_size=64):
    poems = prepareData(path)
    poems = traditional2simplified(poems)
    vocab = prepareVocab(poems)
    fullData = word2idx(poems, vocab)
    fullData = fullData[:3000]
    train_data = []
    for d in range(int(len(fullData)/batch_size)):
        inp = np.zeros((batch_size,len(fullData[d])))
        y = np.zeros((batch_size,len(fullData[d])))
        for j in range(batch_size):
            data = fullData[d*batch_size+j]
            inp[j] = np.asarray(data)
            data = np.asarray(data)    
            y[j][:-1],y[j][-1] = data[1:],data[0]
        train_data.append((inp.transpose(1,0),y.transpose(1,0)))
    return train_data, vocab

def softmax(x):
    
    if x.shape[1] == 1:
        tmp = max(x)
        x -= tmp
        return (np.exp(x)/np.sum(np.exp(x)))
    
    else:
        tmp = np.max(x)
        x -= np.ones(x.shape) * tmp
        vec = np.dot(np.exp(x.T),np.ones((x.shape[0],1)))
        vec = 1/(vec * np.ones((1,x.shape[0])))
        return np.multiply(np.exp(np.mat(x)),vec.T)

#model 

class lstmPoemNumpy():

    def __init__(self,input_dim, hidden_dim, vocab_size):
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size,input_dim)
        
        self.Wf = np.random.random((hidden_dim, input_dim + hidden_dim))*np.sqrt(2/(hidden_dim*2+input_dim))
        self.Wi = np.random.random((hidden_dim, input_dim + hidden_dim))*np.sqrt(2/(hidden_dim*2+input_dim))
        self.Wc = np.random.random((hidden_dim, input_dim + hidden_dim))*np.sqrt(2/(hidden_dim*2+input_dim))
        self.Wo = np.random.random((hidden_dim, input_dim + hidden_dim))*np.sqrt(2/(hidden_dim*2+input_dim))
        
        self.Bf = np.random.random((hidden_dim, 1))
        self.Bi = np.random.random((hidden_dim, 1))
        self.Bc = np.random.random((hidden_dim, 1))
        self.Bo = np.random.random((hidden_dim, 1))
        
        self.w = np.random.random((vocab_size, hidden_dim))*np.sqrt(2/(hidden_dim+vocab_size))
        self.b = np.random.random((vocab_size, 1))
        
    
        
    def forward(self, inp, batch_size, init_states=None):
        
        seq_len = inp.shape[0]
        h = np.zeros((seq_len+1,self.hidden_dim,batch_size))
        c = np.zeros((seq_len+1,self.hidden_dim,batch_size))
        if init_states != None:
            h[-1], c[-1] = init_states
        
        f = np.zeros((seq_len+1, self.hidden_dim, batch_size))
        i = np.zeros((seq_len+1, self.hidden_dim, batch_size))
        o = np.zeros((seq_len+1, self.hidden_dim, batch_size))
        g = np.zeros((seq_len+1, self.hidden_dim, batch_size))
        
        z = np.zeros((seq_len+1, self.hidden_dim + self.input_dim, batch_size))
        
        target = []
        for t in range(seq_len):
            tmp = self.embedding(torch.tensor(inp[t]).long()).permute(1,0).detach().numpy()
            
            z[t] = np.concatenate((tmp,h[t-1]),0)
            f[t] = sigmoid(self.Wf.dot(z[t]) + self.Bf)
            i[t] = sigmoid(self.Wi.dot(z[t]) + self.Bi) 
            o[t] = sigmoid(self.Wo.dot(z[t]) + self.Bo) 
            g[t] = tanh(self.Wc.dot(z[t]) + self.Bc) 
            c[t] = np.multiply(f[t], c[t-1]) + np.multiply(i[t], g[t])
            h[t] = np.multiply(o[t], tanh(c[t]))
            
            a = self.w.dot(h[t])+self.b
            target.append(softmax(a))
        target = np.array(target).reshape(seq_len,self.vocab_size,batch_size)
        return target,(z,f,i,g,o,h,c)
    

    
    def predict(self, x):
        target = self.forward(x)
        pre_y = np.argmax(target.reshape(len(x), -1).numpy(), axis=1)         
        return pre_y
    
    def loss(self, score, y):
        loss = 0        
        for j in range(score.shape[2]):
            cc = np.array([score[i][int(y[i][j])][j] for i in range(len(y))])
            pre_yi = np.log(cc)
            loss -= np.sum(pre_yi)
        ave_loss = (loss / len(y))/y.shape[1]
        return ave_loss
    
    def init_grad(self,batch_size):
        dw = np.zeros(self.Wf.shape)
        db = np.zeros((self.Bf.shape[0],batch_size))
        return dw,db
    
    def bptt(self, target, y, state, rate):
        
        dWf, dBf = self.init_grad(target.shape[-1])
        dWi, dBi = self.init_grad(target.shape[-1])
        dWo, dBo = self.init_grad(target.shape[-1])
        dWc, dBc = self.init_grad(target.shape[-1])
        
        dw,db = np.zeros((self.w.shape[0],self.w.shape[1])),np.zeros((self.b.shape[0],target.shape[-1]))
        deltaCt = np.zeros((self.hidden_dim,target.shape[-1]))
        
        (z,f,i,g,o,h,c) = state
        dt = target
        
        for t in range(len(y)):
            for j in range(len(y[t])):
                dt[t][int(y[t][j])][j] -= 1
        
        for tt in range(len(y)):
            t = int(len(y) - tt - 1)
            inpZ = z[t].transpose((1,0))
            dw += dt[t].dot(h[t].transpose((1,0)))
            db += dt[t]
            dht = self.w.T.dot(dt[t])
            deltaOt = dht * tanh(c[t])
            deltaCt += dht * o[t] * (1-tanh(c[t])**2)
            deltaCt1 = deltaCt * f[t]
            deltaIt = deltaCt * g[t]
            deltaFt = deltaCt * c[t-1]
            deltaGt = deltaCt * i[t]
            dg = deltaGt * (1-g[t]**2)
            di = deltaIt * i[t] * (1-i[t])
            df = deltaFt * f[t] * (1-f[t])
            do = deltaOt * o[t] * (1-o[t])
            dWc += dg.dot(inpZ)
            dBc += dg
            dWi += di.dot(inpZ)
            dBi += di
            dWf += df.dot(inpZ)
            dBf += df
            dWo += do.dot(inpZ)
            dBo += do
            
            deltaCt = deltaCt1
            deltaCt1 = np.zeros((self.hidden_dim,target.shape[-1]))
        
        
        rate = rate/target.shape[-1]
        self.Wf -= rate * dWf
        tmp = np.sum(dBf,axis = 1)
        self.Bf -= rate * tmp.reshape((len(tmp),1))
        self.Wi -= rate * dWi
        tmp = np.sum(dBi,axis = 1)
        self.Bi -= rate * tmp.reshape((len(tmp),1))
        self.Wo -= rate * dWo
        tmp = np.sum(dBo,axis = 1)
        self.Bo -= rate * tmp.reshape((len(tmp),1))
        self.Wc -= rate * dWc
        tmp = np.sum(dBc,axis = 1)
        self.Bc -= rate * tmp.reshape((len(tmp),1))
        self.w -= rate * dw
        tmp = np.sum(db,axis = 1)
        self.b -= rate * tmp.reshape((len(tmp),1))

#generation

def generatePoem(model, vocab, start_word):
    start_id = np.array([[vocab.to_index(start_word)]])
    result = []
    result.append(start_word)
    input_ = start_id
    for i in range(19):
        output, _ = model.forward(input_, 1)
        output = output.reshape(-1)
        top20 = output.argsort()[::-1][0:20]
#         print(output)
        
        index = top20[random.randint(0,19)]
        w = vocab.to_word(index)
        result.append(w)
        input_ = np.array([[vocab.to_index(w)]])
        
    return result

#train
def trainNumpy(epochs=8, rate=0.001, batch_size=64):
    train_data, vocab = preprocess()
    model = lstmPoemNumpy(128, 128, len(vocab))
    generatePoem(model, vocab, "天")
    losses = []
    for e in range(epochs):
        i = 0
        count = 0
        lossSum = 0
        for data, y in train_data:
            score,state = model.forward(data,batch_size)
            loss = model.loss(score,y)
            model.bptt(score,y,state,rate)
            print("Epoch: %d batch:(%d,%d) loss: %.6f" % (e, i * batch_size, (i+1) * batch_size, loss))
            gen_poetry = ''.join(generatePoem(model, vocab, "日"))
            print(gen_poetry)
            i += 1
            lossSum += loss
        
        losses.append(lossSum/i)
    return model, losses


model, loss = trainNumpy()

#loss
import matplotlib.pyplot as plt
def drawPic(y):
    plt.figure()
    y = y.reshape(-1)
    x = np.arange(1,len(y)+1,1)
    plt.plot(x, y)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
    
drawPic(np.array(loss))