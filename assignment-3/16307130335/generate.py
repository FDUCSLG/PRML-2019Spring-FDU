import json
import argparse
from model import LSTM
import numpy as np
from vocabulary import *

def load_model(file):
    with open(file, 'r') as load_f:
        m = json.load(load_f)
        hidden_dim = len(m['whi'])
        input_dim = len(m['wxi'][0])
        vocab_dim = len(m['by'])
        lstm = LSTM(input_dim, hidden_dim, vocab_dim)
        lstm.whc = np.array(m['wha'])
        lstm.wxc = np.array(m['wxa'])
        lstm.bc = np.array(m['ba'])
        lstm.whf = np.array(m['whf'])
        lstm.wxf = np.array(m['wxf'])
        lstm.bf = np.array(m['bf'])
        lstm.whi = np.array(m['whi'])
        lstm.wxi = np.array(m['wxi'])
        lstm.bi = np.array(m['bi'])
        lstm.who = np.array(m['who'])
        lstm.wxo = np.array(m['wxo'])
        lstm.bo = np.array(m['bo'])
        lstm.wy = np.array(m['wy'])
        lstm.by = np.array(m['by'])
    return lstm



def next_word(lstm, x, hist, vocab):
    # x为单个汉字
    index = vocab.to_index(x)
    pre_y, hist = lstm.predict([index], hist)
    y = vocab.to_word(pre_y[0])
    return y, hist

def arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", default="model_29.json")
    args = parser.parse_args()
    return args

def main():
    args = arg()
    lstm = load_model(args.file)
    train_data, dev_data = get_dataset()
    vocab, train_data, dev_data = get_vocabulary(train_data, dev_data)
    while True:
        print("请选择：\n"
              "0: 五言\n"
              "1: 七言\n")
        choice = int(input())
        if choice == 0:
            num = 5
        elif choice == 1:
            num = 7
        print("请输入句数：")
        sent_num = int(input())
        if sent_num > 0:
            print("请输入诗句首字：")
            start_words = str(input())
            poem = start_words
            hist = {'h': 0, 'c': 0}
            for i in range(2, sent_num * 2 * num + 1):
                y, hist = next_word(lstm, start_words, hist, vocab)
                poem += y
                if i % (2 * num) == 0:
                    poem += '。\n '
                elif i % num == 0:
                    poem += '，'
        print("生成诗句:\n" , poem)


if __name__=="__main__":
    main()