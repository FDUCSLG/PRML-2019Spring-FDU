import collections
import torch

from data_process import *


poetry_list = []
file_path = './../handout/tangshi.txt'
file = open(file_path, encoding='utf-8')
poetries = file.read().strip().split('\n\n')
poetry_list = [poetry.replace('\n', '') for poetry in poetries]
n_poetries = len(poetry_list)

word_list = []
for poetry in poetry_list:
    word_list.extend([word for word in poetry])
counter = collections.Counter(word_list)
sorted_word_list = sorted(counter.items(), key=lambda x : x[1], reverse=True)
word_list = [x[0] for x in sorted_word_list]
word_list.append('E')
n_words = len(word_list)
poetry_list = [poetry + 'E' for poetry in poetry_list]


rnn = torch.load('./rnn_3_2_948.97589_RMSprop.pt')
max_length = 100

def test(start_word='日'):
    input = create_input_tensor(start_word + 'E', word_list)
    hidden = rnn.init_hidden()
    output_poetry = start_word
    for i in range(max_length):
        output, hidden = rnn(input, hidden)
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
print(test('红'))
print(test('山'))
print(test('夜'))
print(test('湖'))
print(test('海'))
print(test('月'))