import string
import re
import random
import numpy as np
from matplotlib import pyplot as plt

#Part 2
def Get_Vocabulary(data):
    text_list = []
    for text in data:
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub('[' + string.whitespace + '\u200b]+', ' ', text)
        text = text.strip().lower()
        list = text.split(' ')
        text_list.append(list)
    all_word = [x for j in text_list for x in j]
    print("Total Number of Words:", len(all_word))
    frequency = {}
    for word in all_word:
        if word not in frequency:
            frequency.update({word: 0})
        else:
            frequency[word] += 1
    frequent_word = [j for j in frequency if frequency[j] >= 10]
    word_list = sorted(frequent_word)
    print("Size of Vocabulary:", len(word_list))
    vocab = {word: i for i, word in enumerate(word_list)}
    print("Get Vocabulary!")
    return vocab

def Preprocess(data, target, K, vocab):
    M = len(vocab)
    text_list = []
    for text in data:
        text = text.translate(str.maketrans("","", string.punctuation))
        text = re.sub('[' + string.whitespace + '\u200b]+', ' ', text)
        text = text.strip().lower()
        list = text.split(' ')
        text_list.append(list)
    multi_hot = []
    for text in text_list:
        tmp = [0 for i in range(0, M)]
        for word in text:
            if word in vocab:
                tmp[vocab[word]] = 1
        multi_hot.append(tmp)
    one_hot = np.zeros((len(data), K))
    for i, category in enumerate(target):
        one_hot[i, category] = 1
    print("Preprocessing OK!")
    return np.mat(multi_hot), np.mat(one_hot)

def softmax(x):
    x = x - np.mean(x, axis = 1)
    return np.exp(x)/(np.sum(np.exp(x), axis = 1))

def Cross_Entropy_Loss(X, y, W, b, lamb):
    N = X.shape[0]
    y_hat = softmax(X * W + b.T)
    Loss = - np.trace((y * np.log(y_hat).T)) / N + lamb * np.sum(np.multiply(W, W))
    return Loss

def Gradient_Calculator(X, y, W, b, lamb):
    y_hat = softmax(X * W + b.T)
    Wgradient = - X.T * (y - y_hat) / X.shape[0] + 2 * lamb * W
    bgradient = (- np.sum((y - y_hat), axis = 0) / X.shape[0]).T
    return Wgradient, bgradient

def Gradient_Checker(data, target, lamb = 0.1, error = 0.01):
    N, M, K = data.shape[0], data.shape[1], target.shape[1]
    for cnt in range(0, 100):
        W = np.mat(np.zeros((M, K)))
        b = np.mat(np.zeros((K, 1)))
        old_loss = Cross_Entropy_Loss(data, target, W, b, lamb)
        i, j = random.randint(0, M - 1), random.randint(0, K - 1)
        Wgrad, bgrad = Gradient_Calculator(data, target, W, b, lamb)
        Wgrad = Wgrad[i, j]
        W[i, j] += 0.001
        new_loss = Cross_Entropy_Loss(data, target, W, b, lamb)
        approx_grad = (new_loss - old_loss) / 0.001
        if (abs(Wgrad - approx_grad) > error):
            print("Error for (%d, %d)\n\tWgrad:%f\n\tapprox_Wgrad:%f\n\t"%(i, j, Wgrad, approx_grad))
            break
        else:
            print("Good Job for (%d, %d)\n\tWgrad:%f\n\tapprox_Wgrad:%f\n\t"%(i, j, Wgrad, approx_grad))

def Logistic_Regression(data, target, alpha = 0.25, epsilon = 1e-4, lamb = 1e-3, MAXITER = 400, batch = 1):
    N, M, K = data.shape[0], data.shape[1], target.shape[1]
    W = np.mat(np.zeros((M, K)))
    b = np.mat(np.zeros((K, 1)))
    new_Loss = Cross_Entropy_Loss(data, target, W, b, lamb)
    Loss_Record = [new_Loss]
    iter = 0
    while True:
        old_Loss = new_Loss
        iter += 1
        for i in range(0, N, batch):
            Wgradient, bgradient = Gradient_Calculator(data[i:min(i+batch, N)], target[i:min(i+batch, N)], W, b, lamb)
            W -= alpha * Wgradient
            b -= alpha * bgradient
        new_Loss = Cross_Entropy_Loss(data, target, W, b, lamb)
        Loss_Record.append(new_Loss)
        print("Loss at iteration %d: %f"%(iter, new_Loss))
        if old_Loss - new_Loss < epsilon or iter >= MAXITER:
            break
        tot = np.hstack((data, target))
        np.random.shuffle(tot)
        data = tot[0:N, 0:M].copy()
        target = tot[0:N, M:M+K].copy()

    #plt.xlabel("Iteration No.")
    #plt.ylabel("Loss Function")
    #plt.plot(np.linspace(0, iter, len(Loss_Record)), Loss_Record)
    #plt.show()
    return W, b, iter

def Classification_Accuracy(X, target, W, b):
    N = X.shape[0]
    y_hat = softmax(X * W + b.T)
    return np.sum(np.argmax(y_hat, axis = 1).flatten() == target) / N