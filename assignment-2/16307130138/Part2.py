# -*- coding: utf-8 -*-
import os
os.sys.path.append('../')
from handout import get_text_classification_datasets as get_text
import numpy as np
import matplotlib.pyplot as plt
import string
import time
import math

def combine_whitespace(s):
    return s.split()
    
class Part2:
    def __init__(self,epoch:int=300,learning_rate:float=0.1,lamda:float=0.0001,batch_size:int=-1):
        self.dataset_train,self.dataset_test = get_text()
        self.train_size = len(self.dataset_train.data)
        self.test_size = len(self.dataset_test.data)
        self.class_num = len(self.dataset_train.target_names)
        self.min_count = 10
        self.lamda = lamda
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.batch_size = batch_size

        self.word_dict = None
        self.dict_size = None #vec-size
        self.key_order = None
        self.train_multi_hot = None
        print("Logistic Model")
        print("the size of the train set:{0}".format(self.train_size))

    def tokenize(self):
        for i in range(self.train_size):
            text = self.dataset_train.data[i]
            newtext=""
            for c in text:
                if c not in string.punctuation:
                    newtext += c
                else:
                    newtext += ' '
            newtext = combine_whitespace(newtext.lower())
            self.dataset_train.data[i] = newtext
        
        for i in range(self.test_size):
            text = self.dataset_test.data[i]
            newtext=""
            for c in text:
                if c not in string.punctuation:
                    newtext += c
                else:
                    newtext += ' '
            newtext = combine_whitespace(newtext.lower())
            self.dataset_test.data[i] = newtext
    
    def get_word_dict(self):
        self.word_dict = {}
        for text in self.dataset_train.data:
            for word in text:
                if word in self.word_dict.keys():
                    self.word_dict[word] += 1
                else:
                    self.word_dict[word] = 1
        
        word_list = list(self.word_dict.keys())
        for word in word_list:
            if self.word_dict[word] < self.min_count:
                del self.word_dict[word]
        print("the size of the valid dictionary:{}".format(len(self.word_dict.keys())) )
        self.dict_size = len(self.word_dict)
        
        self.key_order = {}
        sortkey = sorted(self.word_dict.keys())
        for i in range(self.dict_size):
            key = sortkey[i]
            self.key_order[key] = i
    
    def get_multi_hot_vector(self,data):
        multi_hot = []
        for text in data:
            vec=[0 for i in range(self.vec_size)]
            # vec[0]=1
            for word in text:
                if word in self.word_dict.keys():
                    #vec[1+self.key_order[word]]=1
                    vec[self.key_order[word]]=1
            multi_hot.append(vec)
        return np.array(multi_hot)
        
    def preprocessing(self):
        self.tokenize()
        self.get_word_dict()
        self.vec_size = self.dict_size
        
        self.train_data = self.get_multi_hot_vector(self.dataset_train.data)
        print("shape of the train set:{}".format(self.train_data.shape) )

        self.target = np.zeros((self.class_num, self.train_size))
        for n in range(self.train_size):
            yn = self.dataset_train.target[n]
            self.target[yn][n] = 1
        print("shape of the train target:{}".format(self.target.shape))
        print(self.target)
        if(self.batch_size == -1):
            self.batch_size = self.target.shape[1]
    
    def softmax(self,x):
        shiftx = x - np.max(x,axis = 0,keepdims = True)
        sum_exp = np.sum(np.exp(shiftx),axis = 0, keepdims = True)
        self.logp = shiftx - np.log(sum_exp)
        return np.exp(self.logp)

    def forward(self):
        self.terminate = True
        self.Loss = 0.0
        self.S = self.W.dot(self.train_data.T) + self.b
        self.P = self.softmax(self.S)
        self.Loss = -np.sum( self.target * self.logp ) / self.train_size
        print("Loss-origin:{}".format(self.Loss))
        regularization = self.lamda * (np.linalg.norm(self.W) ** 2)
        print("Loss-regularization:{}".format(regularization))
        self.Loss += regularization
        
    def batch_gradient(self,X,y,W,b,lamda):
        punc = 2*lamda*W
        # print("shape 1",W.dot(X.T).shape)
        # print("shape of b:",b.shape)
        # print("multi:",W.dot(X.T).shape)
        # print("multi+b:",(W.dot(X.T)+b).shape)
        p = self.softmax( W.dot(X.T) + b )
        # print("b:",b.shape)
        # print("p:",p.shape)
        # print("y:",y.shape)
        tmp = p - y
        # print("X:",X.shape)
        # print("tmp:",tmp.shape)
        # print("punc:",punc.shape)
        wgradient = (tmp.dot(X) )/X.shape[0] + punc
        bgradient = (np.sum(tmp,axis = 1,keepdims = True) / X.shape[0] )
        return wgradient,bgradient

    def backward(self):
        detaw,detab = self.batch_gradient(self.train_data,self.target,self.W,self.b,self.lamda)
        self.W -= self.learning_rate * detaw
        self.b -= self.learning_rate * detab

    def para_initial(self):
        #initialize W and b
        self.W = np.zeros((self.class_num,self.vec_size))
        self.b = np.zeros((self.class_num,1))
        #self.b = np.zeros((self.class_num,self.train_size))
        print("shape of W:{}".format(self.W.shape))
        print("shape of b:{}".format(self.b.shape))
        print("class_num:{}".format(self.class_num))
        print("train_size:{}".format(self.train_size))
        self.S = np.random.random((self.class_num,self.train_size))
        print("shape of S:{}".format(self.S.shape) )
        self.P = np.random.random((self.class_num,self.train_size))
        print("shape of P:{}".format(self.P.shape) )
        self.Loss = 100000000.0

    def logistic_model(self):
        self.preprocessing()
        self.para_initial()
        count = 0
        loss_list = []
        while(1):
            self.forward()
            loss_list.append(self.Loss)
            print("{}th round,Loss:{}".format(count,self.Loss))
            if (count>=self.epoch):
                break
            self.backward()
            count += 1
        #plot
        dbx = [i for i in range(len(loss_list))]
        dby = loss_list
        plt.plot(dbx,dby,'r',label = "learning_rata={}".format(self.learning_rate))
        print("accuracy:{}".format(self.model_varify()))
        plt.title("Loss Curve")
        plt.xlabel("Round")
        plt.ylabel("Loss")
        plt.legend(loc = "upper right")
        plt.show()
    
    def test_learning_rate(self):
        self.preprocessing()
        for i in range(5):
            self.learning_rate = 1/10**i
            print("Learning Rate of this round:{}".format(self.learning_rate))
            self.para_initial()
            count = 0
            loss_list = []
            while(1):
                self.forward()
                loss_list.append(self.Loss)
                print("{}th round,Loss:{}".format(count,self.Loss))
                if (count >= self.epoch):
                    break
                self.backward()
                count += 1
            #plot
            dbx = [i for i in range(len(loss_list))]
            dby = loss_list
            plt.plot(dbx,dby,label = "learning_rate={}".format(self.learning_rate))
        plt.title("Loss Curve")
        plt.xlabel("Round")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")
        plt.show() 

    def test_lamda(self):
        self.preprocessing()
        for i in range(5):
            self.lamda = 1/10**i
            print("Regularization degree(lambda) of this round:{}".format(self.lamda))
            self.para_initial()
            count = 0
            loss_list = []
            while(1):
                self.forward()
                loss_list.append(self.Loss)
                print("{}th round,Loss:{}".format(count,self.Loss))
                if (count >= self.epoch):
                    break
                self.backward()
                count += 1
            #plot
            dbx = [i for i in range(len(loss_list))]
            dby = loss_list
            plt.plot(dbx,dby,label = "lambda={}".format(self.lamda))
            acc = self.model_varify()
            print("{}th accuracy:{}".format(i,acc))
        plt.title("Loss Curve")
        plt.xlabel("Round")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")
        plt.show() 
    
    def backward_SGD(self,X,y):
        detaw,detab = self.batch_gradient(X,y,self.W,self.b,self.lamda)
        self.W -= self.learning_rate * detaw
        self.b -= self.learning_rate * detab

    def Logistic_SGD(self):
        self.preprocessing()
        self.para_initial()
        start = time.clock()
        count = 0
        loss_list = []
        random_sample = np.random.randint(0,self.train_size,size = [1,self.epoch * self.batch_size])
        while(1):
            self.forward()
            loss_list.append(self.Loss)
            print("{}th round,Loss:{}".format(count,self.Loss))
            if (count >= self.epoch):
                break
            left = count*self.batch_size
            right = min( (count+1)*(self.batch_size),self.epoch*self.batch_size )
            X = self.train_data[random_sample[0,left:right ],: ].reshape(self.batch_size,-1)
            y = self.target[:,random_sample[0,left:right] ].reshape(-1,self.batch_size)
            self.backward_SGD(X,y)
            count += 1
        #plot
        end = time.clock()
        print("time cost:{}".format(end-start))
        dbx = [i for i in range(len(loss_list))]
        dby = loss_list
        plt.plot(dbx,dby,'r',label = "epoch={}".format(self.epoch))
        print("accuracy:{}".format(self.model_varify()) )
        plt.title("Loss Curve")
        plt.xlabel("Round")
        plt.ylabel("Loss")
        plt.legend(loc = "best")
        plt.show()

    def test_batch_size(self):
        self.preprocessing()
        size_lst = [1,10,50,100,500,1000]
        start = time.clock()
        for i in size_lst:
            self.batch_size = i
            print("batch_size of this round:{}".format(self.batch_size))
            self.para_initial()
            count = 0
            loss_list = []
            count = 0
            random_sample = np.random.randint(0,self.train_size,size = [1,self.epoch * self.batch_size])
            while(1):
                self.forward()
                loss_list.append(self.Loss)
                print("{}th round,Loss:{}".format(count,self.Loss))
                if (count >= self.epoch):
                    break
                left = count*self.batch_size
                right = min( (count+1)*(self.batch_size),self.epoch*self.batch_size )
                X = self.train_data[random_sample[0,left:right ],: ].reshape(self.batch_size,-1)
                y = self.target[:,random_sample[0,left:right] ].reshape(-1,self.batch_size)
                self.backward_SGD(X,y)
                count += 1
            #plot
            dbx = [i for i in range(len(loss_list))]
            dby = loss_list
            plt.plot(dbx,dby,label = "batch_size={}".format(self.batch_size))
            acc = self.model_varify()
            print("{}th accuracy:{}".format(i,acc))
        end = time.clock()
        plt.title("Loss Curve")
        plt.xlabel("Round")
        plt.ylabel("Loss")
        plt.legend(loc="best")
        plt.show() 
    
    def test_batch_size2(self):
        self.preprocessing()
        total_number = 2000
        size_lst = [1,10,50,100,500,1000]
        final_loss = []
        total_time = []
        for i in size_lst:
            self.batch_size = i
            self.epoch = (int)(total_number/i)
            print("batch_size of this round:{}".format(self.batch_size))
            self.para_initial()
            count = 0
            loss_list = []
            count = 0
            random_sample = np.random.randint(0,self.train_size,size = [1,self.epoch * self.batch_size])
            start = time.clock()
            while(1):
                self.forward()
                loss_list.append(self.Loss)
                print("{}th round,Loss:{}".format(count,self.Loss))
                if (count >= self.epoch):
                    final_loss.append(self.Loss)
                    break
                left = count*self.batch_size
                right = min( (count+1)*(self.batch_size),self.epoch*self.batch_size )
                X = self.train_data[random_sample[0,left:right ],: ].reshape(self.batch_size,-1)
                y = self.target[:,random_sample[0,left:right] ].reshape(-1,self.batch_size)
                self.backward_SGD(X,y)
                count += 1
            #plot
            end = time.clock()
            total_time.append(end-start)

            dbx = [i for i in range(len(loss_list))]
            dby = loss_list
            plt.plot(dbx,dby,label = "batch_size={} ##{}".format(self.batch_size,end-start))
            acc = self.model_varify()
            print("{}th accuracy:{}".format(i,acc))

        print(size_lst)
        print(total_time)
        print(final_loss)
        plt.title("Loss Curve (batch_size * epoch = {})".format(total_number))
        plt.xlabel("Round")
        plt.ylabel("Loss")
        plt.legend(loc="best")
        plt.show()
    
    def model_varify(self):
        self.test_data = self.get_multi_hot_vector(self.dataset_test.data)
        print("shape of the test set:{}".format(self.test_data.shape) )
        self.test_target = np.zeros((self.class_num, self.test_size))
        for n in range(self.test_size):
            yn = self.dataset_test.target[n]
            self.test_target[yn][n] = 1
        print("shape of the test target:{}".format(self.test_target.shape))
        print(self.test_target)

        p = self.softmax( self.W.dot(self.test_data.T) + self.b )
        print(p)
        maxp = np.max(p,axis = 0,keepdims = True)
        print(maxp)
        #print(maxp.shape)
        #pp = np.where(p==np.max(p,axis = 0,keepdims = True))
        ppl = []
        right = 0
        wrong = 0
        for i in range(self.test_size):
            m = maxp[0][i]
            #print("m:",m)
            mj = 0
            for j in range(self.class_num):
                if (abs(p[j][i]-m)<1e-7):
                    mj = j
                    ppl.append(j)
                    break
            if (self.test_target[mj][i]==1):
                right +=1
            else :
                # print(self.test_target[:,i])
                # print(mj)
                wrong += 1

        # print(ppl)
        
        # for i in pp[1]:
        #     #print(i)
        #     #print(pp[0][i])
        #     if(self.test_target[ pp[0][i] ,i ] == 1):
        #         right += 1
        #     else:
        #         wrong += 1
        
        print(right)
        print(wrong)
        accuracy = right/(right+wrong)
        print("accuracy:"+str(accuracy) )
        return accuracy

    def gradient_check(self,round):
        theta = 1e-7
        threshold = 1e-7
        self.preprocessing()
        self.para_initial()
        # X = self.train_data
        self.forward()
        
        #random_sample = np.random.randint(0,self.train_size,size = [1,10])
        for r in range(round):
            #sample_id = random_sample[0,r]
            self.W = np.random.random((self.class_num,self.vec_size))
            self.b = np.random.random((self.class_num,1))
            gw,gb = self.batch_gradient(self.train_data,self.target,self.W,self.b,self.lamda)
            countb = 0
            count = 0
            for i in range(self.class_num):
                for j in range(self.vec_size):
                    #gw,gd = self.batch_gradient(self.train_data,self.target,self.W,self.b,self.lamda)
                    gwij = gw[i,j]
                    self.W[i,j] -= theta
                    self.forward()
                    l1 = self.Loss

                    self.W[i,j] += 2*theta
                    self.forward()
                    l2 = self.Loss
                    
                    dwij = (l2-l1)/(2*theta)
                    print("gwij:{} \ndwij:{}".format(gwij,dwij) )
                    count = 0
                    if(abs(dwij-gwij)<threshold):
                        count+=1
                        #print("Yes")
                    else:
                        return False
                        print("NoOOOOOOOOOOOOOOOOOOO!")
                
                gbi = gb[i]
                self.b[i] -= theta
                self.forward()
                l1 = self.Loss
                self.b[i] += 2*theta
                self.forward()
                l2 = self.Loss
                dbi = (l2-l1)/(2*theta)
                if(abs(dbi-gbi)<threshold):
                    countb+=1
                    #print("Yes")
                else:
                    return False
                    print("NoOOOOOOOOOOOOOOOOOOO!")
            print(count + countb)
        return True