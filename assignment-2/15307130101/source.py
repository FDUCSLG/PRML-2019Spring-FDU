import os
os.sys.path.append('..')
# use the above line of code to surpass the top module barrier
from handout import gen_linear_seperatable_2d_2c_dataset, get_text_classification_datasets
import numpy as np
import matplotlib.pyplot as plt
import math

d = get_linear_seperatable_2d_2c_dataset()
d.plot(plt).show()

## PART 1   
### least square model
def least_square_fld(dataset,plt):
    # Function estimates the LDA parameters
    ## pass train data only
    ## In version 1,S_B is calculated according lecture
    ## S_B=(m2-m1)(m2-m1).T
    def estimate_params(train_data):
        data=train_data
        # group data by label column
        train_t=np.array([v for index,v in enumerate(data.X) if data.y[index]==True])
        train_f=np.array([v for index,v in enumerate(data.X) if data.y[index]==False])

        # calculate means for each class
        means = {}
        means[True]=train_t.mean(axis=0) ##([meanx,meany])
        means[False]=train_f.mean(axis=0)
        

        # calculate the overall mean of all the data
        overall_mean = data.X.mean(axis = 0)
        ##print("overall_mean",overall_mean)

        #  version_1, different from teacher's
        # calculate between class covariance matrix    
        # S_B = (m2-m1)(m2-m1).T
        S_B = np.zeros((data.X.shape[1], data.X.shape[1]))
        S_B+=np.outer((means[True]-means[False]),(means[True]-means[False]))

        # calculate within class covariance matrix
        # S_W = \sigma{S_i}
        # S_i = \sigma{(x - m_i) (x - m_i).T}
        S_W = np.zeros(S_B.shape) 
        for index,l in enumerate(data.y):
            ele_v= data.X[index]-means[l]
            S_W=np.add(np.outer(ele_v,ele_v),S_W)

        # objective : find eigenvalue, eigenvector pairs for inv(S_W).S_B
        mat = np.dot(np.linalg.pinv(S_W), S_B)
        eigvals, eigvecs = np.linalg.eig(mat)
        eiglist = [(eigvals[i], eigvecs[:, i]) for i in range(len(eigvals))]

        # sort the eigvals in decreasing order
        eiglist = sorted(eiglist, key = lambda x : x[0], reverse = True)

        # take the first num_dims eigvectors  //j(w)
        ##w = np.array([eiglist[i][1] for i in range(self.num_dims)])
        w = np.array(eiglist[0][1])

        print("w is ",w)
        return w
    
    ## true is yellow color
    def calculate_accuracy(test_data,w):
        tot=len(test_data.y)
        right_def=0.0
        ambi=0.0
        for index, ele in enumerate(test_data.X):
            tmp=ele[0]*w[0]+w[1]
            if tmp==ele[1]:
                ambi=ambi+1
                continue
            if tmp>ele[1] and test_data.y[index] or tmp<ele[1] and not test_data.y[index]:
                right_def=right_def+1
        accuracy_def=right_def/tot
        accuracy_max=(right_def+ambi)/tot
        print("minimum accuracy(miss judge all ambiguous data): ",str(accuracy_def*100),"%")
        print("maximum accuracy(judge all ambiguous data right): ",str(accuracy_max*100),"%")
        return accuracy_def,accuracy_max
        

    def show_data(plt,w,data,title):
        plt.plot([-1.5,1.5],[-1.5*w[0]+w[1],1.5*w[0]+w[1]])
        plt.title(title)
        data.plot(plt).show()
    ds_train,ds_test=dataset.split_dataset()
    # estimate the LDA parameters
    w=estimate_params(ds_train)
    show_data(plt,w,ds_train,"Picture 1 Training Data")
    show_data(plt,w,ds_test,"Picture 2 Test Data")
    calculate_accuracy(ds_test,w)
    
    
least_square_fld(d,plt)


### perceptron model
def Perceptron(dataset,epoch,batch_size,l_rate,plt):
    def sigmoid_activation(x):
        # compute and return the sigmoid activation value for a
        # given input value
        return 1.0 / (1 + np.exp(-x))

    def next_batch(X, y, batchSize):
        # loop over our dataset `X` in mini-batches of size `batchSize`
        for i in np.arange(0, X.shape[0], batchSize):
            # yield a tuple of the current batched data and labels
            yield (X[i:i + batchSize], y[i:i + batchSize])

    
    # Calculate accuracy percentage
    def accuracy_metric(actual, X_t,W):
        Y = (-W[0] - (W[1] * X_t[:,1])) / W[2]
        
        predicted=[False if ele[2] > Y[index]  else True for index,ele in enumerate(X_t)]
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0
        
        
        
    
    ##draw data and decision line
    def draw(plt,X,y,W,title):
        # compute the line of best fit by setting the sigmoid function
        # to 0 and solving for X2 in terms of X1
        Y = (-W[0] - (W[1] * X)) / W[2]
        # plot the original data along with our line of best fit
        plt.figure()
        plt.scatter(X[:, 1], X[:, 2], marker="o", c=y)
        #plt.plot(X, Y, "r-")
        plt.plot([-1.5,1.5],[-(W[0]+W[1]*(-1.5))/W[2],-(W[0]+W[1]*(1.5))/W[2]],"r-")
        plt.title(title)
        
        '''
    def draw_old(plt,X,y,W,title):
        # compute the line of best fit by setting the sigmoid function
        # to 0 and solving for X2 in terms of X1
        Y = (-W[0] - (W[1] * X)) / W[2]
        # plot the original data along with our line of best fit
        plt.figure()
        plt.scatter(X[:, 1], X[:, 2], marker="o", c=y)
        plt.plot(X, Y, "r-")
        plt.title(title)
        '''
        
    def draw_taining_loss(plt,epoch,lossHistory):
        # construct a figure that plots the loss over time
        fig = plt.figure()
        plt.plot(np.arange(0, epoch+1), lossHistory)
        fig.suptitle("Picture 5 Training Loss")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss")
        plt.show()
        
        
    
    # insert a column of 1's as the first entry in the feature
    # vector -- this is a little trick that allows us to treat
    # the bias as a trainable parameter *within* the weight matrix
    # rather than an entirely separate variable
    ds_train, ds_test=dataset.split_dataset()
    X=ds_train.X
    y=ds_train.y
    X = np.c_[np.ones((X.shape[0])), X]

    # initialize our weight matrix such it has the same number of
    # columns as our input features
    print("[INFO] starting training...")
    W = np.random.uniform(size=(X.shape[1],))

    # initialize a list to store the loss value for each epoch
    lossHistory = []

    # loop over the desired number of epochs
    for epoch in np.arange(0, epoch):
        # initialize the total loss for the epoch
        epochLoss = []

        # loop over our data in batches
        for (batchX, batchY) in next_batch(X, y, batch_size):
            # take the dot product between our current batch of
            # features and weight matrix `W`, then pass this value
            # through the sigmoid activation function
            preds = sigmoid_activation(batchX.dot(W))

            # now that we have our predictions, we need to determine
            # our `error`, which is the difference between our predictions
            # and the true values
            error = preds - batchY

            # given our `error`, we can compute the total loss value on
            # the batch as the sum of squared loss
            loss = np.sum(error ** 2)
            epochLoss.append(loss)

            # the gradient update is therefore the dot product between
            # the transpose of our current batch and the error on the
            # # batch
            gradient = batchX.T.dot(error) / batchX.shape[0]

            # use the gradient computed on the current batch to take
            # a "step" in the correct direction
            W += -l_rate * gradient

        # update our loss history list by taking the average loss
        # across all batches
        lossHistory.append(np.average(epochLoss))

        
    X_t=np.c_[np.ones((ds_test.X.shape[0])), ds_test.X]
    print("accuarcy is ",accuracy_metric(ds_test.y,X_t,W),"%")
    
    draw(plt,X,y,W,"Picture 3 Training Data")

    draw(plt,X_t,ds_test.y,W,"Picture 4 Test Data")
    
    draw_taining_loss(plt,epoch,lossHistory)
    
    

Perceptron(d,100,10,0.01,plt)


## PART 2

import string
from handout import get_linear_seperatable_2d_2c_dataset,get_text_classification_datasets
import numpy as np
import matplotlib.pyplot as plt
import math

categories,dataset_train_p2,dataset_test_p2=get_text_classification_datasets()

def document_embeding(dataset_train):
    ## build list of string from train data
    list_of_string=[]
    tokenized_sentences=[]
    trainslator=str.maketrans(string.punctuation,' '*len(string.punctuation))
    for ele in dataset_train.data:
        # remove repeated data in one line
        sentence=list(set(ele.lower().translate(trainslator).replace(string.whitespace,' ').split()))
        tokenized_sentences.append(sentence)
        list_of_string+=sentence

    list_of_string=list(set(list_of_string))
    list_of_string.sort()
    ##build vocabulary 
    vocabulary={}
    for index,ele in enumerate(list_of_string):
        vocabulary[ele]=index
        #print(ele)
    
    ## build multi-hot vec
    multi_hot_v=[]
    for ele in tokenized_sentences:
        v=[0 for i in range(len(vocabulary))]
        for e in ele:
            v[vocabulary[e]]=1
        multi_hot_v.append(v)
        #print (v)
    return np.mat(multi_hot_v),vocabulary
multi_hot_v,vocabulary=document_embeding(dataset_train_p2)

## K is the number of kinds also number of columns
def categories_embedding(target,K):
    one_hot = np.zeros((len(target), K))
    for i, column in enumerate(target):
        one_hot[i][column] = 1
    return np.mat(one_hot)

def softmax(x):
    return np.exp(x)/(np.sum(np.exp(x), axis = 1))

class Pa_xia:
    def __init__(self,W,X,y_n,b,lambd):
        self.y_hat=softmax(X * W.T + b.T)
        #print("w",W)
        self.X=X
        self.y_n=y_n
        self.lambd=lambd
        self.W=W
        
    def pa_xia_w(self):
        #print("gradient w, x.shape[0]: ",self.X.shape[0])
        #print("y_hat :",self.y_hat)
        gradient = - self.X.T * (self.y_n - self.y_hat) / self.X.shape[0]+(2 * self.lambd * self.W).T
        return gradient
    
    def pa_xia_b(self):
        #print("gradient b, self.x.shape[0]",self.X.shape[0])
        gradient = (- np.sum((self.y_n - self.y_hat), axis = 0) / self.X.shape[0]).T
        return gradient


# W k*d
# target must be in vector
def cross_entropy_loss(X,target,W,b):
    #print("w",str(W.shape))
    #print("b",str(b.shape))
    #print("x",str(X.shape))
    y_hat=np.dot(W,X.T)+b ##z 一列
    y_hat=np.exp(y_hat)
    divider=np.sum(y_hat,axis=1)
    y_hat=y_hat/divider
    y_hat=np.log(y_hat)
    N=X.shape[0]
    loss=-np.trace(target*y_hat)/N
    return loss
    
    
def logistic_regression(X,target,l_rate,lambd,batch_size=1,iteration=100,epsilon=0.0001):
    N=X.shape[0]
    d=X.shape[1]
    K=4
    W=np.mat(0.1*np.random.rand(K,d))
    b=np.mat(0.1*np.random.rand(K,1))
    target_v=categories_embedding(target,K)
    new_loss=cross_entropy_loss(X, target_v, W, b)
    loss_history=[]
    loss_history.append(new_loss)
    while True:
        old_Loss = new_loss
        iteration-=1
        for i in range(0, N, batch_size):
            last_entry=min(i+batch_size,N)
            p=Pa_xia(W,np.mat(X[i:last_entry]),np.mat(target_v[i:last_entry]),np.mat(b),lambd)
            w_gradient=p.pa_xia_w()
            b_gradient = p.pa_xia_b()
            #print("b",b_gradient)
            #print("w",w_gradient)
            delta_w=l_rate * w_gradient
            W=W-delta_w.T
            delta_b=l_rate * b_gradient
            b=b-delta_b
            
        new_Loss = cross_entropy_loss(X, target_v, W, b)
        #print("new loss",new_loss)
        loss_history.append(new_Loss)
        print("Loss at iteration %d: %f"%(iteration, new_Loss))
        if old_Loss - new_Loss < epsilon or iteration < 0:
            break
        ##X_T = np.hstack((X, target))
        ##np.random.shuffle(X_T)
        ##X = X_T[0:N, 0:d].copy()
        ##target = X_T[0:N, d].copy()
        ##target_v=categories_embedding(target,K)
    return W,b,loss_history
W,B,loss_history_1=logistic_regression(X=multi_hot_v,target=dataset_train_p2.target,l_rate=1.5,lambd=0,batch_size=100,iteration=200,epsilon=0.01)
#X,target,l_rate,lambd,batch_size=1,iteration=100,epsilon=0.0001
print(loss_history_1)


x_axis=[i for i in range(len(loss_history_1))]
plt.plot(x_axis,loss_history_1)
plt.xlabel("#epoch")
plt.title("Picture 6 trainning losss")

W,B,loss_history_2=logistic_regression(X=multi_hot_v,target=dataset_train_p2.target,l_rate=1.5,lambd=0,batch_size=50,iteration=200,epsilon=0.01)
print(loss_history_2)

x_axis=[i for i in range(len(loss_history_1))]
l1=plt.plot(x_axis,loss_history_1,label = 'batch=100')
l2=plt.plot(x_axis,loss_history_2,label = 'batch=50', color = 'red')
plt.xlabel("#epoch")
plt.title("Picture 7 trainning losss comparison")
plt.legend()

def vectorizing(dataset_test,vocabulary):
    ## build list of string from test data
    list_of_string=[]
    tokenized_sentences=[]
    trainslator=str.maketrans(string.punctuation,' '*len(string.punctuation))
    for ele in dataset_test.data:
        # remove repeated data in one line
        sentence=list(set(ele.lower().translate(trainslator).replace(string.whitespace,' ').split()))
        tokenized_sentences.append(sentence)
        list_of_string+=sentence

    ## build multi-hot vec
    multi_hot_v=[]
    for ele in tokenized_sentences:
        v=[0 for i in range(len(vocabulary))]
        for e in ele:
            if e in vocabulary.keys():
                v[vocabulary[e]]=1
        multi_hot_v.append(v)
        #print (v)
    return np.mat(multi_hot_v)
        
test_multi_hot_v=vectorizing(dataset_test_p2,vocabulary)

def Accuracy(X, target, W, b):
    N = X.shape[0]
    y_hat = softmax(X * W.T + b.T)
    return np.sum(np.argmax(y_hat, axis = 1).flatten() == target) / N
a=Accuracy(test_multi_hot_v,dataset_test_p2.target,W,B)

print("test accuracy: ",a)