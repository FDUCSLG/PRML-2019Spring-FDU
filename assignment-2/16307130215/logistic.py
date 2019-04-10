import os,string,argparse,re
os.sys.path.append('..')
# use the above line of code to surpass the top module barrier
from handout import get_text_classification_datasets
import numpy as np
import matplotlib.pyplot as plt

dataset_train, dataset_test = get_text_classification_datasets()
target_nums = len(dataset_train.target_names)
np.random.seed(233) #设置随机数种子

def train(X_train,Y_train,X_valid,Y_valid,optimizer="bgd",batch_size=10,alpha=1e-1,beta=1e-4):
    assert batch_size>0
    
    W = np.zeros((X_train.shape[1],Y_train.shape[1]))
    b = np.zeros((Y_train.shape[1]))
    
    best_acc = 0
    best_W = np.zeros_like(W)
    best_b = np.zeros_like(b)
    LOSS=[]
    for epoch in range(2000): #epoch
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_train = X_train[indices]
        Y_train = Y_train[indices] #进行shuffle
        i = 0 
        while i < X_train.shape[0]: #step
            X = X_train[i:i+batch_size]
            Y = Y_train[i:i+batch_size]
            i += batch_size
            scores = X.dot(W) + b
            scores = scores - np.max(scores,axis=1).reshape(-1,1)
            softmax_scores = np.exp(scores)/np.sum(np.exp(scores),axis=1).reshape(-1,1)
            loss = (- np.sum(Y*np.log(softmax_scores)))/X.shape[0] + beta*np.sum(W*W)
            dW = X.T.dot(softmax_scores-Y) / X.shape[0]+ 2*beta*W
            db = np.sum(softmax_scores-Y,axis=0) / X.shape[0]            
            W = W - alpha*dW
            b = b - alpha*db
            # print(loss)
            LOSS.append(loss)
        
        train_acc,valid_acc = calculate_accuracy(W,b,X_train,Y_train,X_valid,Y_valid,epoch)
        print("epoch={}: train_acc={}\t\tvalid_acc={}".format(epoch+1,train_acc,valid_acc))
        if valid_acc>=best_acc: #若验证集的正确率有所提升，更新最佳的W，b
            best_W = W
            best_b = b

        if (len(LOSS)>=2 and abs(LOSS[-1]-LOSS[-2])<1e-4) or LOSS[-1]<1e-4: #当两次loss的差小于给定的阈值
                                                                            #或者loss小于给定阈值时，训练终止
            plt.xlabel("step")
            plt.ylabel("loss")
            plt.title("optimizer = {}\nbatch size = {}\nlearning rate = {}".format(optimizer,batch_size,alpha))
            plt.plot(LOSS)
            plt.show()
            break

    return best_W,best_b
    
def calculate_accuracy(W,b,X_train,Y_train,X_valid,Y_valid,epoch):#计算训练集与验证集的正确率
    train_scores = X_train.dot(W) + b
    train_predict = np.argmax(train_scores,axis=1)
    train_real = np.argmax(Y_train,axis=1)
    train_acc = np.sum(train_predict==train_real)/len(X_train)
    
    valid_scores = X_valid.dot(W) + b
    valid_predict = np.argmax(valid_scores,axis=1)
    valid_real = np.argmax(Y_valid,axis=1)
    valid_acc = np.sum(valid_predict==valid_real)/len(X_valid)

    return train_acc,valid_acc

def test(W,b):
    test_data = Split_data(dataset_test.data)
    test_X,_ = Get_X_Y(test_data,dataset_test.target)
    scores = test_X.dot(W)
    predict = np.argmax(scores,axis=1)
    test_acc = np.sum(predict==dataset_test.target)/len(test_data)
    return test_acc

def Split_data(data): #切割为单词
    data = [ re.sub('[%s]' % re.escape(string.punctuation), '', item) for item in data] #忽略标点符号
    data = [ re.sub('[%s]' % re.escape(string.whitespace), ' ', item) for item in data] #将空格、换行符等空白替换为空格
    data = [item.lower().split(' ') for item in data]                                   #转换为小写
    return data

def Generate_vocabulary(data): #通过一系列单词构建字典
    voca = dict()
    for item in data:
        for t in item:
            if t in voca.keys():
                voca[t]+=1
            else:
                voca[t]=1
    
    voca = [item for item in voca if voca[item]>=10] #忽略出现次数不足10的
    voca = {item:index for index,item in enumerate(voca)}
    # print(len(voca))
    # exit()
    return voca

def Get_X_Y(data,target):
    num_data = len(data)
    X = np.zeros((num_data,len(voca)))
    for i in range(num_data):
        for item in data[i]:
            if item in voca.keys():
                j = voca[item]
                X[i][j]=1

    Y = np.zeros((num_data,target_nums))
    for i,j in enumerate(target):
        Y[i][j] = 1
    
    return X,Y

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-optimizer', type=str,choices=["fbgd","bgd","sgd"], default="bgd", help='优化器')
    parser.add_argument('-learning_rate',type=float, default=1e-1, help='学习率')
    parser.add_argument('-batch_size',type=int,default=10)
    args = parser.parse_args()

    train_data_raw = Split_data(dataset_train.data)
    voca = Generate_vocabulary(train_data_raw) #构建字典
    X_raw,Y_raw = Get_X_Y(train_data_raw,dataset_train.target) #获取原始的X,Y
    num_data = X_raw.shape[0] #总的训练样本数
    indices = np.arange(num_data)
    np.random.shuffle(indices)
    X_raw = X_raw[indices]
    Y_raw = Y_raw[indices] #对原始的X,Y进行shuffle
   
    data_slice = int(num_data/10)
    X_valid = X_raw[:data_slice]
    Y_valid = Y_raw[:data_slice]
    X_train = X_raw[data_slice:]
    Y_train = Y_raw[data_slice:] #切割为训练集与测试集（测试集占1/10）

    if args.optimizer == "fbgd":
        batch_size = X_train.shape[0]
    elif args.optimizer == "sgd":
        batch_size = 1
    else:
        batch_size = args.batch_size

    W,b = train(X_train,Y_train,X_valid,Y_valid,optimizer=args.optimizer,batch_size=batch_size,alpha=args.learning_rate,beta=1e-4) #训练
    test_acc = test(W,b) #测试

    print("\nThe accuracy of test_set is {}%".format(test_acc*100))