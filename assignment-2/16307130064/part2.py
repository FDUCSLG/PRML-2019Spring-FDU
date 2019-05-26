import os
os.sys.path.append('..')
# use the above line of code to surpass the top module barrier
from handout import *
import string
import random
random.seed(100)

class Vocab:
    def __init__(self,mini=10):
        self.w2i={}
        self.w2c={}
        self.mini=mini
        
    def add(self,sentence):
        for i in sentence.split(" "):
            if len(i)==0:
                continue
            if i not in self.w2c:
                self.w2c[i]=1
            else: 
                self.w2c[i]+=1
                if self.w2c[i]>=self.mini and i not in self.w2i:
                    self.w2i[i]=len(self.w2i)
                    #print(i,len(self.w2i))
    def toi(self,sentence):
        nw=[self.w2i[x] for x in sentence.split(" ") if len(x)>0 and x in self.w2i]
        return nw
        
def deal(raw):
    nw=[]
    for r,p in zip(list(raw.data),list(raw.target)):
        #print(r)
        for sp in string.punctuation:
            r=r.replace(sp,"")
        for sp in string.whitespace:
            r=r.replace(sp," ")
        nw.append((r.lower(),p))
        #print(r.lower().split(" "))
        #break
    return nw  

def vectorize(y,m):
    assert(y<m)
    e=np.zeros(m)
    e[y]=1.0
    return e

def softmax(X):
    a=np.exp(X)
    return a/np.sum(a,axis=0,keepdims=True)

def transform(data,m,ot):
    n=len(data)
    In=np.zeros((m,n))
    Out=np.zeros((ot,n))
    for i in range(n):
        cur=data[i]
        #In[0][i]=1.0
        for word in cur[0]:
            In[word][i]=1.0
        Out[:,i]=vectorize(int(cur[1]),ot)
    return In,Out

def prepare():
    train,test=get_text_classification_datasets()
    train=deal(train)
    test=deal(test)
    vocX=Vocab(10)
    for x,y in train:
        vocX.add(x)
    print(len(vocX.w2i))
    #print(train[1][0])
    train=[(vocX.toi(x),y) for x,y in train]
    test=[(vocX.toi(x),y) for x,y in test]
    #print(train[1][0])
    return train,test,vocX

class Regression:
    def __init__(self,in_size,out_size):
        self.w=0.005*np.random.randn(out_size,in_size)
        self.b=np.zeros((out_size,1))
        self.n=in_size
        self.m=out_size
        
    def train(self,training,validation=None,mini=20,eta=0.01,epoch=10,lamda=0):
        self.w=np.zeros((self.m,self.n))#0.005*(2*np.random.randn(self.m,self.n)-1)
        self.b=np.zeros((self.m,1))
        l=[]
        v=[] if validation is not None else None
        for i in range(epoch):
            random.shuffle(training)
            batches=[training[k:k+mini] for k in range(0,len(training),mini)]
            for batch in batches:
                X,Y=transform(batch,self.n,self.m)
                h=np.dot(self.w,X)+self.b
                #print(np.mean(h,axis=0))
                a=softmax(h-np.mean(h,axis=0))
                loss=-np.sum(Y*np.nan_to_num(np.log(a)))/len(batch)+lamda*np.sum(np.square(self.w))
                delta=np.dot(a-Y,X.transpose())/len(batch)+2*lamda*self.w
                self.w-=eta*delta
                self.b-=eta*np.mean(a-Y,axis=1,keepdims=True)
            print(i,loss,end=" ")
            l.append(loss)
            if validation is not None:
                a=self.accuracy(validation)
                print(a)
                v.append(a)
            else: print()
        return l,v
        
    def accuracy(self,test):
        n=len(test)
        batches=[test[k:k+1000] for k in range(0,n,1000)]
        s=0.0
        for batch in batches:
            X,Y=transform(batch,self.n,self.m)
            a=softmax(np.dot(self.w,X)+self.b)
            s+=np.sum(np.argmax(a,axis=0)==np.argmax(Y,axis=0))       
        return 1.0*s/n

def start(batch=20,epoch=100,lr=1e-4,lamda=0,skip_dev=False,full=False):        
    train,test,voc=prepare()
    valid=None
    random.shuffle(train)
    if not skip_dev:
        train_size=int(len(train)*0.9)
        train,valid=train[:train_size],train[train_size:]
    if full:
        batch=len(train)
        
    r=Regression(len(voc.w2i),4)
    loss,Va=r.train(train,valid,mini=batch,eta=lr,epoch=epoch,lamda=lamda)
    acc=r.accuracy(test)
    return loss,Va,acc

    
