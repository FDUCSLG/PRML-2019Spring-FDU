import numpy as np
from fastNLP import DataSet
from fastNLP import Vocabulary
from matplotlib import pyplot as plt
import json
import re
import math
import os

def read_vectors(path, topn=0):  # read top n word vectors, i.e. top is 10000
    lines_num, dim = 0, 0
    vectors = {}
    with open(path, encoding='utf-8', errors='ignore') as f:
        first_line = True
        for line in f:
            if first_line:
                first_line = False
                dim = int(line.rstrip().split()[1])
                continue
            lines_num += 1
            tokens = line.rstrip().split(' ')
            vectors[tokens[0]] = np.asarray([float(x) for x in tokens[1:]])
            if topn != 0 and lines_num >= topn:
                break
    return vectors, dim


def update_vectors(word, delta, alpha=0.01):
    vectors[word] = vectors[word] - alpha * delta.reshape(-1)


def get_vector(word):
    try:
        vec = vectors[word]
    except:
        #print("%s not in vectors!"%word)
        vectors[word] = np.random.rand(dim)
        vec = vectors[word]
    return vec


filedir = os.getcwd() + "/dataset"
filenames = os.listdir(filedir)
data = []
for i in filenames:
    filepath = filedir + '/' + i
    f = open(filepath, 'rb')
    dat = json.load(f)
    data.append(dat)
fil = re.compile(r'[\\s+\\.\\!\\/_,$%^*(+\\\"\')]+ | [+——()?【】“”！，。？、~ @  # ￥%……&*（）]+')
poems = []
for i in data:
    for j in i:
        tmp = ""
        for k in j['paragraphs']:
            tmp += k
        tmp = re.sub(fil, '', tmp)
        if len(tmp) < 20:
            continue
        elif len(tmp) < 60:
            tmp = tmp.rjust(60)
        poems.append(tmp[:60])
print(len(poems))
poems = poems[:10000]
sep = math.ceil(len(poems) * 0.8)
train_data = poems[:sep]
test_data = poems[sep:]

# Preprocess
sl = 60
# dataSet = DataSet.read_csv("../handout/tangshi.txt", headers=["raw_sentence"])
# dataSet.drop(lambda x: len(x['raw_sentence']) != sl)
# test_data, train_data = dataSet.split(0.8)

vocab = Vocabulary(min_freq=1)
# dataSet.apply(lambda x: [vocab.add(word) for word in x['raw_sentence']])
for i in poems:
    for word in i:
        vocab.add(word)
vocab.add("EOS")
vocab.build_vocab()
print(len(vocab))
m, n, dim = len(vocab), 64, 64

vectors = []
for i in range(m):
    vectors.append(np.random.rand(dim))

#m = output_layer_size
#n = hidden_layer_size
sigma = 20
W_A = np.random.normal(0, 1 / sigma, (m, n + 1))
W_C = np.random.normal(0, 1 / sigma, (n, n + dim + 1))
W_f = np.random.normal(0, 1 / sigma, (n, n + dim + 1))
W_i = np.random.normal(0, 1 / sigma, (n, n + dim + 1))
W_o = np.random.normal(0, 1 / sigma, (n, n + dim + 1))
# ---------------------------------------------------------------------------


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    x -= np.max(x)
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def perplexity():
    avg = 0
    for i in range(0, len(test_data)):
        log_perp = 0
        sentence = test_data[i]
        h = np.zeros((n, 1))
        C = np.zeros((n, 1))
        for t in range(0, sl):
            x = np.array([get_vector(vocab[sentence[t]])]).T
            z = np.concatenate((h, x, np.ones((1, 1))), axis=0)
            f = sigmoid(np.dot(W_f, z))
            o = sigmoid(np.dot(W_o, z))
            i = sigmoid(np.dot(W_i, z))
            tildeC = np.tanh(np.dot(W_C, z))
            C = f * C + i * tildeC
            h = o * np.tanh(C)
            a = np.dot(W_A, np.concatenate((h, np.ones((1, 1))), axis=0))
            a -= np.max(a)
            y = softmax(a)
            if t < sl - 1:
                word = vocab[sentence[t + 1]]
            else:
                word = vocab["EOS"]
            log_perp -= np.log(y[word])
        avg += log_perp / sl
    avg /= len(test_data)
    return avg


def Train(alpha=0.01, rho=0.9, batch_size=1, beta=0.99, eps=1e-8):
    global W_A, W_C, W_f, W_o, W_i
    lst_perp, epoch, iter = perplexity(), 0, 0
    print("Perplexity of Epoch %d on test_set is %.5f" % (epoch, lst_perp))
    train_size = len(train_data)
    train_Perp_Record = []
    Perp_Record = [lst_perp]
    W_A_Delta = np.zeros_like(W_A)
    W_C_Delta = np.zeros_like(W_C)
    W_f_Delta = np.zeros_like(W_f)
    W_o_Delta = np.zeros_like(W_o)
    W_i_Delta = np.zeros_like(W_i)
    # '''
    # RMSprop
    G_W_A = np.zeros_like(W_A)
    G_W_C = np.zeros_like(W_C)
    G_W_f = np.zeros_like(W_f)
    G_W_o = np.zeros_like(W_o)
    G_W_i = np.zeros_like(W_i)
    # '''
    while True:
        try:
            epoch += 1
            train_loss = 0
            iter = 0
            for num in range(0, train_size, batch_size):
                iter += 1
                size = min(train_size - num, batch_size)
                Loss = 0
                h = [np.zeros((n, size))]
                C = [np.zeros((n, size))]
                x, z, f, o, i, tildeC, a, y = [], [], [], [], [], [], [], []
                for t in range(0, sl):
                    word_list = []
                    for j in range(num, num + size):
                        word_list.append(get_vector(vocab[train_data[j][t]]))
                    x.append(np.array(word_list).T)
                    z.append(np.concatenate((h[t], x[t], np.ones((1, size))), axis=0))
                    f.append(sigmoid(np.dot(W_f, z[t])))
                    o.append(sigmoid(np.dot(W_o, z[t])))
                    i.append(sigmoid(np.dot(W_i, z[t])))
                    tildeC.append(np.tanh(np.dot(W_C, z[t])))
                    C.append(f[t] * C[t] + i[t] * tildeC[t])
                    h.append(o[t] * np.tanh(C[t + 1]))
                    a.append(np.dot(W_A, np.concatenate((h[t + 1], np.ones((1, size))), axis=0)))
                    a[t] -= np.max(a[t])
                    y.append(softmax(a[t]))
                    for j in range(num, num + size):
                        if t < sl - 1:
                            word = vocab[train_data[j][t + 1]]
                        else:
                            word = vocab["EOS"]
                        Loss = Loss - np.log(y[t][word, j - num])
                Loss = Loss / size / sl
                train_loss += Loss * size
                # print(Loss)

                W_A_grad = np.zeros_like(W_A)
                W_C_grad = np.zeros_like(W_C)
                W_f_grad = np.zeros_like(W_f)
                W_o_grad = np.zeros_like(W_o)
                W_i_grad = np.zeros_like(W_i)
                for j in range(num, num + size):
                    t = sl - 1
                    diff_h = np.zeros_like(h[t][:, [j-num]])
                    diff_c = np.zeros_like(C[t][:, [j-num]])
                    while t >= 0:
                        if t < sl - 1:
                            word = vocab[train_data[j][t + 1]]
                        else:
                            word = vocab["EOS"]

                        # Calculate top_diff_h
                        tmp = np.zeros((m, 1))
                        tmp[word] += 1
                        tmp = (y[t][:, [j - num]] - tmp) / sl
                        W_A_grad += np.outer(tmp, np.concatenate((h[t + 1][:, [j-num]], np.ones((1, 1))), axis=0))
                        tmp = np.dot(W_A.T, tmp)

                        # Update diff_h
                        diff_h += tmp[:-1]
                        # Back Propagation
                        s = 1 - np.tanh(C[t + 1][:, [j-num]]) ** 2
                        ds = o[t][:, [j-num]] * s * diff_h + diff_c
                        do = np.tanh(C[t + 1][:, [j-num]]) * diff_h
                        di = tildeC[t][:, [j-num]] * ds
                        dc = i[t][:, [j-num]] * ds
                        df = C[t][:, [j-num]] * ds

                        d_i_input = (1 - i[t][:, [j-num]]) * i[t][:, [j-num]] * di
                        d_f_input = (1 - f[t][:, [j-num]]) * f[t][:, [j-num]] * df
                        d_o_input = (1 - o[t][:, [j-num]]) * o[t][:, [j-num]] * do
                        d_c_input = (1 - tildeC[t][:, [j-num]] ** 2) * dc
                        W_i_grad += np.outer(d_i_input, z[t][:, [j-num]])
                        W_f_grad += np.outer(d_f_input, z[t][:, [j-num]])
                        W_o_grad += np.outer(d_o_input, z[t][:, [j-num]])
                        W_C_grad += np.outer(d_c_input, z[t][:, [j-num]])

                        dxc = np.zeros_like(z[t][:, [j-num]])
                        dxc += np.dot(W_i.T, d_i_input)
                        dxc += np.dot(W_f.T, d_f_input)
                        dxc += np.dot(W_o.T, d_o_input)
                        dxc += np.dot(W_C.T, d_c_input)

                        diff_c = f[t][:, [j-num]] * ds
                        diff_h = dxc[:n]
                        diff_x = dxc[n:-1]
                        update_vectors(vocab[train_data[j][t]], diff_x, alpha)
                        t -= 1
                '''
                # Momentum
                W_A_Delta = rho * W_A_Delta - alpha * W_A_grad / size
                W_C_Delta = rho * W_C_Delta - alpha * W_C_grad / size
                W_f_Delta = rho * W_f_Delta - alpha * W_f_grad / size
                W_o_Delta = rho * W_o_Delta - alpha * W_o_grad / size
                W_i_Delta = rho * W_i_Delta - alpha * W_i_grad / size
                '''

                # '''
                # RMSprop
                G_W_A = beta * G_W_A + (1 - beta) * W_A_grad * W_A_grad
                G_W_C = beta * G_W_C + (1 - beta) * W_C_grad * W_C_grad
                G_W_f = beta * G_W_f + (1 - beta) * W_f_grad * W_f_grad
                G_W_o = beta * G_W_o + (1 - beta) * W_o_grad * W_o_grad
                G_W_i = beta * G_W_i + (1 - beta) * W_i_grad * W_i_grad

                W_A_Delta = - alpha / np.sqrt(G_W_A + eps) * W_A_grad
                W_C_Delta = - alpha / np.sqrt(G_W_C + eps) * W_C_grad
                W_f_Delta = - alpha / np.sqrt(G_W_f + eps) * W_f_grad
                W_o_Delta = - alpha / np.sqrt(G_W_o + eps) * W_o_grad
                W_i_Delta = - alpha / np.sqrt(G_W_i + eps) * W_i_grad
                # '''


                W_A += W_A_Delta
                W_C += W_C_Delta
                W_f += W_f_Delta
                W_i += W_i_Delta
                W_o += W_o_Delta

                if iter % 1000 == 0: print(iter, Loss.item())

            train_loss /= train_size
            #now_perp = perplexity(data=train_data)
            print("Perplexity of Epoch %d on training_set is %.5f" % (epoch, train_loss))
            now_perp = perplexity()
            train_Perp_Record.append(train_loss)
            print("Perplexity of Epoch %d is %.5f" % (epoch, now_perp))
            Perp_Record.append(now_perp)
            if epoch >= 100: break
            lst_perp = now_perp
        except KeyboardInterrupt:
            break

    plt.xlabel("Epoch No.")
    plt.ylabel("Loss Function")
    plt.plot(np.linspace(0, len(Perp_Record), len(Perp_Record)), Perp_Record)
    plt.show()

    plt.xlabel("Epoch No.")
    plt.ylabel("Loss Function on training set")
    plt.plot(np.linspace(0, len(train_Perp_Record), len(train_Perp_Record)), train_Perp_Record)
    plt.show()

# ------------------------------------------------------------------------------------

def Output(word):
    fout = open("%s.txt" % word, "w")
    fout.write(word)
    x = np.array([get_vector(vocab[word])]).T
    h = np.zeros((n, 1))
    C = np.zeros((n, 1))
    for t in range(0, sl):
        z = np.concatenate((h, x, np.ones((1, 1))), axis=0)
        f = sigmoid(np.dot(W_f, z))
        o = sigmoid(np.dot(W_o, z))
        i = sigmoid(np.dot(W_i, z))
        tildeC = np.tanh(np.dot(W_C, z))
        C = f * C + i * tildeC
        h = o * np.tanh(C)
        a = np.dot(W_A, np.concatenate((h, np.ones((1, 1))), axis=0))
        y = softmax(a)
        new_word = np.random.choice(m, 1, p=y.reshape(-1))[0]
        fout.write(vocab.to_word(new_word))
        x = np.array([get_vector(new_word)]).T
    print("%s Complete!" % word)


if __name__ == '__main__':
    Train()
    Output("日")
    Output("红")
    Output("山")
    Output("夜")
    Output("湖")
    Output("海")
    Output("月")