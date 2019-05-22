from model import LSTM
from vocabulary import *
import numpy as np
import argparse
global step
step = 0


def train(lstm, X, y, learning_rate, epoch_num):
    losses = []
    for epoch in range(epoch_num):
        for i in range(len(y)):
        	lstm.backward(X[i], y[i], learning_rate)
        global step
        step += 1
        loss = lstm.loss(X, y)
        print("step"+str(step)+":loss =" + str(loss))
        losses.append(loss)
        #if len(losses) > 1 and losses[-1] > losses[-2]:
        #    learning_rate *= 0.5
         #   print('decrease learning_rate to', learning_rate)
    return learning_rate, lstm


def generate_model(lstm, name):
    model = {}
    model['whi'] = lstm.whi.tolist()
    model['wxi'] = lstm.wxi.tolist()
    model['who'] = lstm.who.tolist()
    model['wxo'] = lstm.wxo.tolist()
    model['whc'] = lstm.whc.tolist()
    model['wxc'] = lstm.wxc.tolist()
    model['whf'] = lstm.whf.tolist()
    model['wxf'] = lstm.wxf.tolist()
    model['bf'] = lstm.bf.tolist()
    model['bo'] = lstm.bo.tolist()
    model['bc'] = lstm.bc.tolist()
    model['bi'] = lstm.bi.tolist()
    model['wy'] = lstm.wy.tolist()
    model['by'] = lstm.by.tolist()
    with open('model_' + str(name) + '.json', 'w') as f:
        json.dump(model, f)
        print("生成model")

def arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dim", "-i", default=100, type=int)
    parser.add_argument("--hidden_dim", "-l", default=100, type=int)
    parser.add_argument("--embedding_dim", "-e", default=100, type=int)
    parser.add_argument("--epoch_num", "-n", default=2, type=int)
    parser.add_argument("--learning_rate", "-r", default=0.005, type=float)
    parser.add_argument("--output", "-o", default="2019")
    parser.add_argument("--batch", "-b", default=5, type=int)
    args = parser.parse_args()
    return args


def main():
    args = arg()
    train_data, dev_data = get_dataset()
    vocab, train_data, dev_data = get_vocabulary(train_data, dev_data)
    lstm = LSTM(args.input_dim, args.hidden_dim, len(vocab))
    X_train = []
    y_train = []
    for poem in train_data:
        sent = poem['words']
        lenth = len(sent)
        x = sent[0:lenth - 2]
        y = sent[1:lenth - 1]
        if x != [] and y != []:
            X_train.append(list(x))
            y_train.append(list(y))
    num = int(len(train_data) / args.batch)
    learning_rate = args.learning_rate
    for i in range(1, num - 1):
        X = np.array(X_train[args.batch*(i-1):args.batch*i])
        y = np.array(y_train[args.batch*(i-1):args.batch*i])
        learning_rate, lstm = train(lstm, X, y, learning_rate, args.epoch_num)
    generate_model(lstm, args.output)





if __name__ == "__main__":
    main()