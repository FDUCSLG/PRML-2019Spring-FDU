import numpy as np

beta = 1e-4
def Gradient_formula(W,b,X,Y):
    scores = X.dot(W) + b
    scores = scores - np.max(scores,axis=1).reshape(-1,1)
    softmax_scores = np.exp(scores)/np.sum(np.exp(scores),axis=1).reshape(-1,1)
    dW = X.T.dot(softmax_scores-Y) / X.shape[0]+ 2*beta*W
    db = np.sum(softmax_scores-Y,axis=0) / X.shape[0]
    return dW,db

def Get_loss(W,b,X,Y):
    scores = X.dot(W) + b
    scores = scores - np.max(scores,axis=1).reshape(-1,1)
    softmax_scores = np.exp(scores)/np.sum(np.exp(scores),axis=1).reshape(-1,1)
    loss = (- np.sum(Y*np.log(softmax_scores)))/X.shape[0] + beta*np.sum(W*W)
    return loss

def Gradient_numerical(W,b,X,Y):
    delta = 1e-5
    dW = np.zeros_like(W)
    db = np.zeros_like(b)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            loss = Get_loss(W,b,X,Y)
            W[i][j] += delta
            delta_loss = Get_loss(W,b,X,Y)
            dW[i][j] = (delta_loss-loss)/delta
            W[i][j] -= delta
    for i in range(b.shape[0]):
        loss = Get_loss(W,b,X,Y)
        b[i] += delta
        delta_loss = Get_loss(W,b,X,Y)
        db[i] = (delta_loss-loss)/delta
        b[i] -= delta
    
    return dW,db

if __name__ == "__main__":
    N = 100
    M = 100
    target = 5
    X =  np.random.randint(0,2,(N,M))
    Y = np.zeros((N,target))
    indices = np.random.randint(0,target,N)
    Y[range(Y.shape[0]),indices] = 1
    W = np.random.randn(M,target)
    b = np.random.randn(target)

    dW_formula,db_formula = Gradient_formula(W,b,X,Y)
    dW_numerical,db_numerical = Gradient_numerical(W,b,X,Y)
    print(dW_formula-dW_numerical)
    print(db_formula-db_numerical)