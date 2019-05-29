import argparse

import torch
import torch.nn as nn
from torch.autograd import Variable

from generator import *
from models import *
from prepare_data import *

if __name__=='__main__':
    data_path = "data.txt"
    test_data, train_data = import_data(data_path)
    test_data, train_data, idx2word, word2idx = construct_vec(
        test_data, train_data)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default='Tang.pth')
    parser.add_argument("--max_len", "-ml", type=int, default=40)
    parser.add_argument("--start_word", "-sw", type=str)
    parser.add_argument("-t", type=float,default=1)
    parser.add_argument("-k", type=int,default=1)
    parser.add_argument("--hidden_size", "-hs", type=int, default=200)
    parser.add_argument("--input_size", "-is", type=int, default=200)
    arg = parser.parse_args()
    model_path=arg.model
    max_len=arg.max_len
    start_word=arg.start_word
    hidden_size = arg.hidden_size
    input_size = arg.input_size

    model = TangPoemGenerator(input_size, hidden_size, len(idx2word))
    model.load_state_dict(torch.load(model_path,'cpu'))
    result = generate(model, start_word, idx2word, word2idx, max_len,arg.t,arg.k)
    print("".join(result))
