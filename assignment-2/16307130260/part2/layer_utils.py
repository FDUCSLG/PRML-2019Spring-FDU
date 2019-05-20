import numpy as np

def affine_forward(w, x, b):
    # C # of categories, N # of cases, D # of input dimention
    # x (N, D)
    # w (D, C)
    # b (C,)
    # out (N, C)
    out = np.dot(x, w) + b
    cache = (w, x, b)
    return out, cache

def affine_backward(dout, cache):
    # C # of categories, N # of cases, D # of input dimention
    # dout (N, C)
    # dw (D, C)
    # db (C,)
    (w, x, b) = cache
    dw = np.dot(x.T, dout)
    db = np.sum(dout, axis=0)
    return dw, db

def softmax_loss(x, y):
    # C # of categories, N # of cases
    # x (N, C)
    # y (N, C) one-hot array 
    # shift x with max of x
    N = x.shape[0]

    # compute loss
    x_shifted = x - np.max(x, axis=1, keepdims=True)
    sum_expx = np.sum(np.exp(x_shifted), axis=1, keepdims=True)
    log_probs = x_shifted - np.log(sum_expx)
    loss = - np.sum(log_probs * y) / N
    
    # get Gradients for x
    probs = np.exp(log_probs)
    dx = (probs - y) / N
    return loss, dx