import os
os.sys.path.append('..')
# use the above line of code to surpass the top module barrier
from handout import *
import numpy as np
import matplotlib.pyplot as plt
import math
import string

def least_square_model(dataset):
    """
    dataset -> (w,b) which y = wx + b
    """
    N = dataset.X.shape[0]
    O = np.ones(shape = [dataset.X.shape[0],1])
    X = np.concatenate([dataset.X,O],axis=1)
    C = [1.0 if (item) else -1.0 for item in dataset.y]

    X_pinv = np.linalg.pinv(X)
    W = np.dot(X_pinv,C)

    pred = np.sign(np.dot(X,W))
    acc = float((C == pred).mean())
    print(acc)
    return -W[0] / W[1],-W[2] / W[1]

def least_square_model_test_main():
    sampled_data = get_linear_seperatable_2d_2c_dataset()
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1, 1)
    sampled_data.plot(ax1)
    a,b = least_square_model(sampled_data)    
    pointsX = [-2,2]
    pointsY = [item * a + b for item in pointsX]
    ax1.plot(pointsX,pointsY)
    plt.show()

def perceptron_model(dataset,iters = 100):
    """
    dataset -> (w,b) which y = wx + b
    """
    N = dataset.X.shape[0]
    O = np.ones(shape = [dataset.X.shape[0],1])
    X = np.concatenate([dataset.X,O],axis=1)
    W = np.random.randn(3,1)
    C = [1.0 if (item) else -1.0 for item in dataset.y]

    for iter in range(iters):
        x = np.dot(X,W)
        loc_n = np.where(x < 0)[0]
        C_pred = np.ones(N)
        C_pred[loc_n] = -1

        t = np.where(C != C_pred)[0]
        if len(t) == 0:
            break
        W += C[t[0]] * X[t[0], :].reshape((3,1))

    pred = np.sign(np.dot(X,W[:,0]))
    acc = float((C == pred).mean())
    print(acc)
    return -W[0][0] / W[1][0],-W[2][0] / W[1][0]

def perceptron_model_test_main():
    sampled_data = get_linear_seperatable_2d_2c_dataset()
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1, 1)
    sampled_data.plot(ax1)
    a,b = perceptron_model(sampled_data,100)
    pointsX = [-2,2]
    pointsY = [item * a + b for item in pointsX]
    ax1.plot(pointsX,pointsY)
    plt.show()

def clean_data(dataset):
    for i in range(len(dataset.data)):
        dataset.data[i] = dataset.data[i].casefold()
        for charS in string.punctuation:
            dataset.data[i] = dataset.data[i].replace(charS,'')
        for charS in string.whitespace:
            dataset.data[i] = dataset.data[i].replace(charS,' ')

def get_dataset(text_dataset,min_count = 10):
    '''
    return a vocab dict
    '''
    clean_data(text_dataset)
    vocab = {}
    wordLists = []
    for item in text_dataset.data:
        lst = item.split(' ')
        wordLists.append(lst)
        for itemL in lst:
            if itemL == '':
                continue
            if not itemL in vocab:
                vocab[itemL] = 1
            else:
                vocab[itemL] += 1
    vocab_ID = {}
    count = 0
    for key in vocab:
        if vocab[key] >= min_count:
            vocab_ID[key] = count
            count += 1
    
    datas = [np.zeros((1,len(vocab_ID))) for i in range(len(text_dataset.data))]

    for i in range(len(text_dataset.data)):
        for word in wordLists[i]:
            if word in vocab_ID:
                datas[i][0][vocab_ID[word]] = 1

    labels = [np.zeros((1,len(text_dataset.target_names))) for i in range(len(text_dataset.data))]
    for i in range(len(text_dataset.data)):
        labels[i][0][text_dataset.target[i]] = 1

    ret = Dataset(datas,text_dataset.target)
    ret.labels = labels
    return ret,vocab_ID

def get_testset(text_dataset,vocab_ID):
    wordLists = []
    for item in text_dataset.data:
        lst = item.split(' ')
        wordLists.append(lst)
    datas = [np.zeros((1,len(vocab_ID))) for i in range(len(text_dataset.data))]

    for i in range(len(text_dataset.data)):
        for word in wordLists[i]:
            if word in vocab_ID:
                datas[i][0][vocab_ID[word]] = 1

    labels = [np.zeros((1,len(text_dataset.target_names))) for i in range(len(text_dataset.data))]
    for i in range(len(text_dataset.data)):
        labels[i][0][text_dataset.target[i]] = 1

    ret = Dataset(datas,text_dataset.target)
    ret.labels = labels
    return ret

def sigmoid(x):
    sigmoid = 1.0 / (1.0 + np.exp(x))
    return sigmoid

def softmax_and_cross_entropy(x,label,norm = True,norm_bias = False,w = [],l = 1):
    row_max = x.max(axis = 1)
    b_size = row_max.shape[0]
    x = x - row_max.reshape(b_size,1)
    x_exp = np.exp(x)
    x_exp_sum = x_exp.sum(axis=1).reshape(b_size,1)
    softmax = x_exp / x_exp_sum
    
    CE = np.log(softmax) * label
    loss = - CE.sum() / b_size

    if norm:
        if norm_bias:
            loss = loss + l * np.linalg.norm(w)
        else:
            loss = loss + l * np.linalg.norm(w[:,:-1])
    return loss,softmax

def softmax_and_cross_entropy_grad(x,label,last_grad,norm=True,norm_bias = False,w=[],l = 1):
    diffs = x - label
    diffs_sum = diffs#.sum(axis=0)
    grad = diffs_sum * last_grad
    d_W = np.zeros_like(w)
    if norm:
        if norm_bias:
            d_W = 2 * l * w
        else:
            d_w = 2 * l * w
            for row in d_W.shape[0]:
                d_w[row,-1] = 0

    return grad,d_W

def sigmoid_grad(x,last_grad):
    sigmoid = 1.0 / (1.0 + np.exp(x))
    return last_grad * (sigmoid * (1-sigmoid))

def dot_grad(x,last_grad):
    b_size = x.shape[0]
    return np.dot(x.transpose(),last_grad) / b_size

def one_step(X,Y,W,norm = True,norm_bias = False,l = 1,lr = 0.1):
    dot = np.dot(X,W)
    sigmoided = sigmoid(dot)
    loss,softmaxed = softmax_and_cross_entropy(sigmoided,Y,norm=norm,norm_bias=norm_bias,w = W,l = l)
    last,d_W = softmax_and_cross_entropy_grad(softmaxed,Y,loss,norm=norm,norm_bias=norm_bias,w=W,l=l)
    last = sigmoid_grad(dot,last)
    last = dot_grad(X,last)

    d_W = last - d_W
    W += lr * d_W

    return loss

def train_logistic_model(dataset,losses = [],batch_size = 100,max_epoches = 100,lr = 0.1,norm = True,l = 0.1):
    dim = dataset.X[0].shape[1]
    label_types = dataset.labels[0].shape[1]
    W = np.random.randn(dim + 1,label_types)
    #W = np.ones([dim + 1,label_types])
    Xs = dataset.X
    Labels = dataset.labels
    total_size = len(Xs)
    for i in range(len(Xs)):
        Xs[i] = np.concatenate([Xs[i],np.ones(shape=(1,1))],axis = 1)


    _permutations = [i for i in range(total_size)]
    for epoch in range(max_epoches):
        np.random.shuffle(_permutations)
        pos = 0
        steps = int(total_size / batch_size)
        for step in range(steps):
            X_ready = [Xs[_permutations[i]] for i in range(pos,pos + batch_size,1)]
            Y_ready = [Labels[_permutations[i]] for i in range(pos,pos + batch_size,1)]
            pos = pos + batch_size
            X = np.concatenate(X_ready,axis = 0)
            Y = np.concatenate(Y_ready,axis = 0)

            loss = one_step(X,Y,W,norm,l,lr)
            losses.append(loss)
            
    return W

def train_logistic_model_with_average_loss_condition(dataset,losses = [],batch_size = 100,max_epoches = 100,lr = 0.1,norm = True,l = 0.1):
    dim = dataset.X[0].shape[1]
    label_types = dataset.labels[0].shape[1]
    W = np.random.randn(dim + 1,label_types)
    Xs = dataset.X
    Labels = dataset.labels
    total_size = len(Xs)
    for i in range(len(Xs)):
        Xs[i] = np.concatenate([Xs[i],np.ones(shape=(1,1))],axis = 1)

    # 基于loss变化的lr和break控制参数
    loss_window_size = 10
    loss_total = 0
    loss_count = 0
    last_loss = 1000000
    break_eps = 0.01
    min_lr = 0.001
    downscale_ratio = 10

    _permutations = [i for i in range(total_size)]
    for epoch in range(max_epoches):
        # 基于loss变化的lr和break控制过程
        if epoch % loss_window_size == 0 and epoch != 0:
            loss = loss_total / loss_count
            print(loss,last_loss - loss,lr)
            if last_loss - loss < break_eps:
                if lr < min_lr:
                    break
                lr = lr / downscale_ratio
                break_eps = break_eps / downscale_ratio
            last_loss = loss
            loss_total = 0
            loss_count = 0
        
        np.random.shuffle(_permutations)
        pos = 0
        steps = int(total_size / batch_size)
        for step in range(steps):
            X_ready = [Xs[_permutations[i]] for i in range(pos,pos + batch_size,1)]
            Y_ready = [Labels[_permutations[i]] for i in range(pos,pos + batch_size,1)]
            pos = pos + batch_size
            X = np.concatenate(X_ready,axis = 0)
            Y = np.concatenate(Y_ready,axis = 0)

            loss = one_step(X,Y,W,norm,l,lr)
            loss_total += loss
            loss_count += 1
            losses.append(loss)
            
    return W

def evaluate(dataset,W):
    Xs = dataset.X
    Labels = dataset.labels
    total_size = len(Xs)
    if Xs[0].shape[1] != W.shape[0]:
        for i in range(len(Xs)):
            Xs[i] = np.concatenate([Xs[i],np.ones(shape=(1,1))],axis = 1)
    X = np.concatenate(Xs,axis=0)
    Y = np.concatenate(Labels,axis=0)

    dot = np.dot(X,W)
    sigmoided = sigmoid(dot)
    loss,softmax = softmax_and_cross_entropy(sigmoided,Y,norm=False)

    maxes = np.argmax(softmax,axis=1)
    print("acc:",dataset.acc(maxes))

if __name__ == '__main__':
    # 取消注释以下代码以运行感知器模型
#    perceptron_model_test_main()
    # 取消注释以下代码以运行least square model
#    least_square_model_test_main()    
    # 取消注释以下代码以运行logistic分类
    datas_train,datas_test = get_text_classification_datasets()
    train_set,vocab_ID = get_dataset(datas_train)
    loss = []
    W = train_logistic_model_with_average_loss_condition(train_set,losses = loss,max_epoches=10000,batch_size=1,norm=True,lr = 0.1,l = 0.01)
    evaluate(train_set,W)
    test_set = get_testset(datas_test,vocab_ID)
    evaluate(test_set,W)
    plt.plot([i for i in range(len(loss))],loss)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.show()
