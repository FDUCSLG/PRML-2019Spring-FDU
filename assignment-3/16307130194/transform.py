import numpy as np
from langconv import *


data = np.load('tang.npz')
data, ix2word = data['data'], data['ix2word'].item()
# print(data[0])
# print(ix2word[data[0][-3]])

poetry = []
for sentence in data:
    text = ''
    for d in sentence:
        if d < 8290:
            text += ix2word[d]
    poetry.append(text)

# print(poetry[0])
max_length = 0
for poem in poetry:
    if len(poem) > max_length:
        max_length = len(poem)

print('Size:', len(poetry))
print('Max length:', max_length)

with open('tangshi.txt', 'w', encoding='utf-8') as f:
    length = len(poetry)
    for i, line in enumerate(poetry):
        line = Converter('zh-hans').convert(line)
        if i < length - 1:
            f.write(line + '\n\n')
        else:
            f.write(line)
