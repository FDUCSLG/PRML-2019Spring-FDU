import collections
import pickle
import torch

from data_process import *


word_list = pickle.load(open('./word_list', 'rb'))
word_list = word_list['word_list']

rnn = torch.load('./rnn_8_310.47545_Adam.pt')
max_length = 100

def test(start_word='日'):
    input = create_input_tensor(start_word + 'E', word_list)
    hidden = rnn.init_hidden()
    output_poetry = start_word
    for i in range(max_length):
        output, hidden = rnn(input.cuda(), hidden)
        topv, topi = output[0].topk(1)
        topi = topi[0]
        word = word_list[int(topi)]
        if word == 'E':
            break
        else:
            output_poetry += word
            input = create_input_tensor(word + 'E', word_list)

    return output_poetry

print(test('日'))
print(test('紅'))
print(test('山'))
print(test('夜'))
print(test('湖'))
print(test('海'))
print(test('月'))